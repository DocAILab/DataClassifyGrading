"""Observation-driven, bounded orchestration for one Tool Call per turn."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
import json
from json import JSONDecodeError
from typing import Any, Iterable, Mapping, Protocol, Sequence

from .catalog import SkillSpec, dynamic_loop_catalog
from .contracts import MetadataRecord, ToolCall
from .runtime import Observation
from .skill_loader import SkillContract, SkillLoader
from .validator import resolve_reference


POLICY_SYSTEM_PROMPT = """You are a database-field classification orchestration policy.
Choose exactly one callable Skill for the current turn. Use only the visible input and named states.
Never invent table metadata, descriptions, values, labels, or ground truth.
Return exactly one JSON object: {"tool":"SKILL_NAME","arguments":{}}.
Do not emit a future Program, Markdown, explanation, plain stop, or extra JSON."""


class ActionError(ValueError):
    """Raised when a policy response is not one valid Tool Call."""


class EarlyStopError(ActionError):
    """Raised when the policy stops before producing a validated final result."""


@dataclass(frozen=True)
class LoopConfig:
    max_turns: int = 8
    max_identical_calls: int = 2
    action_retries: int = 1
    forced_finalize_retries: int = 1
    repetition_retries: int = 1
    recover_ready_finalization: bool = False

    def __post_init__(self) -> None:
        if self.max_turns < 1:
            raise ValueError("max_turns must be at least 1")
        if self.max_identical_calls < 1:
            raise ValueError("max_identical_calls must be at least 1")
        if (
            self.action_retries < 0
            or self.forced_finalize_retries < 0
            or self.repetition_retries < 0
        ):
            raise ValueError("retry limits must be non-negative")


@dataclass(frozen=True)
class LoopEvent:
    kind: str
    turn: int
    data: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "turn": self.turn, "data": dict(self.data)}


@dataclass
class LoopState:
    input: dict[str, str]
    named_state: dict[str, Any] = field(default_factory=dict)
    call_history: list[dict[str, Any]] = field(default_factory=list)
    turns: int = 0
    max_turns: int = 8

    @classmethod
    def from_record(cls, record: MetadataRecord, *, max_turns: int) -> "LoopState":
        return cls(
            input={"domain": record.domain, "field_name": record.field_name},
            max_turns=max_turns,
        )

    @property
    def remaining_turns(self) -> int:
        return max(self.max_turns - self.turns, 0)

    def to_model_dict(self) -> dict[str, Any]:
        return {
            "input": dict(self.input),
            "named_state": dict(self.named_state),
            "call_history": [dict(item) for item in self.call_history],
            "turns": self.turns,
            "remaining_turns": self.remaining_turns,
        }


@dataclass(frozen=True)
class PolicyDecision:
    call: ToolCall
    raw_response: str
    messages: tuple[dict[str, str], ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class LoopTrace:
    input: dict[str, str]
    events: list[LoopEvent] = field(default_factory=list)
    terminated: bool = False
    final_result: dict[str, Any] | None = None
    stop_reason: str = "not_started"
    turns: int = 0
    state: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "input": dict(self.input),
            "events": [event.to_dict() for event in self.events],
            "terminated": self.terminated,
            "final_result": dict(self.final_result) if self.final_result is not None else None,
            "stop_reason": self.stop_reason,
            "turns": self.turns,
            "state": dict(self.state),
        }


class LoopPolicy(Protocol):
    def select_action(
        self,
        record: MetadataRecord,
        state: LoopState,
        callable_skills: Mapping[str, SkillSpec],
        *,
        feedback: str | None = None,
        forced_finalize: bool = False,
    ) -> PolicyDecision: ...


class PrefetchFieldMetadataPolicy:
    """Spend the first normal turn querying metadata, then delegate unchanged."""

    def __init__(self, delegate: LoopPolicy) -> None:
        self.delegate = delegate

    def select_action(
        self,
        record: MetadataRecord,
        state: LoopState,
        callable_skills: Mapping[str, SkillSpec],
        *,
        feedback: str | None = None,
        forced_finalize: bool = False,
    ) -> PolicyDecision:
        if (
            not forced_finalize
            and state.turns == 0
            and not any(item.get("tool") == "GET_FIELD_METADATA" for item in state.call_history)
            and "GET_FIELD_METADATA" in callable_skills
        ):
            call = ToolCall("GET_FIELD_METADATA", {"current_record": "$record"})
            return PolicyDecision(
                call=call,
                raw_response=json.dumps(call.to_dict(), ensure_ascii=False),
                metadata={"fixed_metadata_prefetch": True},
            )
        return self.delegate.select_action(
            record, state, callable_skills,
            feedback=feedback, forced_finalize=forced_finalize,
        )


class PrefetchNeighborFieldsPolicy:
    """Fixed one-time neighbor lookup followed by an unchanged dynamic policy."""

    def __init__(self, delegate: LoopPolicy) -> None:
        self.delegate = delegate

    def select_action(
        self,
        record: MetadataRecord,
        state: LoopState,
        callable_skills: Mapping[str, SkillSpec],
        *,
        feedback: str | None = None,
        forced_finalize: bool = False,
    ) -> PolicyDecision:
        previously_called = any(
            item.get("tool") == "GET_TABLE_FIELDS" for item in state.call_history
        )
        if (
            not forced_finalize
            and not previously_called
            and "table_fields" not in state.named_state
            and "GET_TABLE_FIELDS" in callable_skills
        ):
            call = ToolCall("GET_TABLE_FIELDS", {"current_record": "$record"})
            return PolicyDecision(
                call=call,
                raw_response=json.dumps(call.to_dict(), ensure_ascii=False),
                metadata={"fixed_neighbor_prefetch": True},
            )
        return self.delegate.select_action(
            record, state, callable_skills,
            feedback=feedback, forced_finalize=forced_finalize,
        )


class SkillExecutor(Protocol):
    def execute(
        self,
        call: ToolCall,
        record: MetadataRecord,
        state: Mapping[str, Any],
        contract: SkillContract,
    ) -> Observation: ...


def _decode_one_object(text: str) -> Mapping[str, Any]:
    if not isinstance(text, str) or not text.strip():
        raise ActionError("policy action must be non-empty JSON")
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines.pop()
        cleaned = "\n".join(lines).strip()
    if not cleaned.startswith("{"):
        raise ActionError("policy action must be one JSON object")
    decoder = json.JSONDecoder()
    try:
        value, end = decoder.raw_decode(cleaned)
    except JSONDecodeError as exc:
        raise ActionError(f"policy action is invalid JSON: {exc.msg}") from exc
    if cleaned[end:].strip():
        raise ActionError("policy action contains trailing content")
    if not isinstance(value, Mapping):
        raise ActionError("policy action must be one Tool Call JSON object")
    return value


def parse_policy_action(text: str) -> ToolCall:
    """Parse exactly one Tool Call and reject Programs or plain stop tokens."""

    if isinstance(text, str) and text.strip().lower().rstrip(".!" ) in {
        "stop",
        "done",
        "finish",
        "finished",
        "terminate",
    }:
        raise EarlyStopError("policy stopped without a final result")
    payload = _decode_one_object(text)
    if set(payload) != {"tool", "arguments"}:
        raise ActionError("policy action must be one Tool Call with tool and arguments")
    try:
        return ToolCall.from_mapping(payload)
    except (TypeError, ValueError) as exc:
        raise ActionError(f"invalid Tool Call: {exc}") from exc


def _has_semantic_state(state: Mapping[str, Any]) -> bool:
    return any(
        name in state
        for name in (
            "field_semantics",
            "field_abbreviation_semantics",
            "disambiguated_field_semantics",
            "domain_semantics",
            "field_metadata",
        )
    )


_SEMANTIC_EVIDENCE_KEYS = (
    "domain_semantics",
    "field_abbreviation_semantics",
    "field_semantics",
    "disambiguated_field_semantics",
    "field_description_semantics",
    "table_description_semantics",
    "standard_comparison",
)

_TWO_STAGE_EVIDENCE_KEYS = (
    "domain_semantics",
    "field_abbreviation_semantics",
    "field_semantics",
    "disambiguated_field_semantics",
    "field_metadata",
    "table_fields",
    "field_description_semantics",
    "table_semantics",
    "table_description_semantics",
    "context_field_semantics",
    "metadata_semantics",
    "standard_comparison",
)


def _two_stage_evidence(record: MetadataRecord, state: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "domain": record.domain,
        "field_name": record.field_name,
        "observations": {key: state[key] for key in _TWO_STAGE_EVIDENCE_KEYS if key in state},
    }


def build_final_evidence_state(state: Mapping[str, Any]) -> dict[str, Any] | None:
    """Create the auditable evidence bundle required by the terminal Skill."""

    category = state.get("category")
    level = state.get("grading_level")
    aligned = state.get("aligned_standard_entry")
    candidates = state.get("standard_candidates")
    if (
        not isinstance(category, str)
        or not category
        or not isinstance(level, str)
        or not level
        or not isinstance(aligned, Mapping)
        or not isinstance(candidates, Sequence)
        or isinstance(candidates, (str, bytes))
        or not candidates
    ):
        return None
    semantic_evidence = {
        key: state[key]
        for key in _SEMANTIC_EVIDENCE_KEYS
        if key in state and state[key] not in (None, "")
    }
    return {
        "category": category,
        "grading_level": level,
        "aligned_standard_entry": dict(aligned),
        "standard_candidates": [
            dict(item) if isinstance(item, Mapping) else item for item in candidates
        ],
        "semantic_evidence": semantic_evidence,
    }


def _input_available(name: str, record: MetadataRecord, state: Mapping[str, Any]) -> bool:
    if name == "current_record":
        return True
    if name in {"domain", "field_name"}:
        return record.available(name)
    if name in state:
        return state[name] is not None and state[name] != ""
    if name == "semantic_query":
        return _has_semantic_state(state)
    if name in {"semantic_evidence", "current_evidence", "all_available_evidence"}:
        return bool(state)
    if name == "candidate_standard_categories":
        return bool(state.get("standard_candidates"))
    if name == "aligned_category":
        return bool(state.get("category"))
    if name == "final_evidence_state":
        return isinstance(state.get("final_evidence_state"), Mapping)
    return False


def derive_callable_skills(
    record: MetadataRecord,
    state: LoopState | Mapping[str, Any],
    catalog: Mapping[str, SkillSpec] | None = None,
) -> dict[str, SkillSpec]:
    """Return Skills whose required semantic inputs currently exist."""

    named = state.named_state if isinstance(state, LoopState) else state
    active = catalog or dynamic_loop_catalog()
    return {
        name: spec
        for name, spec in active.items()
        if all(_input_available(required, record, named) for required in spec.required_inputs)
    }


def _catalog_payload(catalog: Mapping[str, SkillSpec]) -> list[dict[str, Any]]:
    return [
        {
            "name": spec.name,
            "required_inputs": list(spec.required_inputs),
            "optional_inputs": list(spec.optional_inputs),
            "produces": list(spec.produces),
            "terminal": spec.terminal,
        }
        for spec in catalog.values()
    ]


def build_policy_messages(
    record: MetadataRecord,
    state: LoopState,
    callable_skills: Mapping[str, SkillSpec],
    *,
    feedback: str | None = None,
    forced_finalize: bool = False,
) -> list[dict[str, str]]:
    """Build a model prompt from the minimum visible input and current states."""

    payload: dict[str, Any] = {
        "task": "Choose the next Skill from the callable catalog or finalize when evidence is sufficient.",
        "current": state.to_model_dict(),
        "callable_skills": _catalog_payload(callable_skills),
        "forced_finalize": forced_finalize,
        "argument_reference_grammar": [
            "$record",
            "$domain",
            "$metadata.field_name",
            "$state.<name>",
        ],
        "output_schema": {"tool": "SKILL_NAME", "arguments": {}},
    }
    if feedback:
        payload["validation_feedback"] = feedback
    return [
        {"role": "system", "content": POLICY_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
    ]


class ScriptedLoopPolicy:
    """Deterministic policy used to test controller behavior without a model."""

    def __init__(self, responses: Iterable[str | ToolCall | Mapping[str, Any]]) -> None:
        self.responses = deque(responses)
        self.seen_states: list[dict[str, Any]] = []
        self.seen_feedback: list[str | None] = []
        self.seen_forced: list[bool] = []

    def select_action(
        self,
        record: MetadataRecord,
        state: LoopState,
        callable_skills: Mapping[str, SkillSpec],
        *,
        feedback: str | None = None,
        forced_finalize: bool = False,
    ) -> PolicyDecision:
        del record, callable_skills
        self.seen_states.append(state.to_model_dict())
        self.seen_feedback.append(feedback)
        self.seen_forced.append(forced_finalize)
        if not self.responses:
            raise ActionError("scripted policy has no response remaining")
        response = self.responses.popleft()
        if isinstance(response, ToolCall):
            call = response
            raw = json.dumps(call.to_dict(), ensure_ascii=False)
        elif isinstance(response, Mapping):
            raw = json.dumps(dict(response), ensure_ascii=False)
            call = parse_policy_action(raw)
        else:
            raw = response
            call = parse_policy_action(raw)
        return PolicyDecision(call=call, raw_response=raw)


class FixedPipelinePolicy:
    """Reference policy that follows the same five evidence steps for every row."""

    def select_action(
        self,
        record: MetadataRecord,
        state: LoopState,
        callable_skills: Mapping[str, SkillSpec],
        *,
        feedback: str | None = None,
        forced_finalize: bool = False,
    ) -> PolicyDecision:
        del record, feedback, forced_finalize
        named = state.named_state
        if "field_semantics" not in named:
            call = ToolCall(
                "INTERPRET_FIELD_NAME",
                {"field_name": "$metadata.field_name", "domain": "$domain"},
            )
        elif "standard_candidates" not in named:
            call = ToolCall("SEARCH_STANDARD", {"semantic_query": "$state.field_semantics"})
        elif "category" not in named:
            call = ToolCall(
                "INFER_CATEGORY",
                {
                    "candidate_standard_categories": "$state.standard_candidates",
                    "all_available_evidence": "$state.field_semantics",
                },
            )
        elif "grading_level" not in named:
            call = ToolCall(
                "INFER_GRADING_LEVEL",
                {
                    "aligned_category": "$state.category",
                    "aligned_standard_entry": "$state.aligned_standard_entry",
                },
            )
        else:
            call = ToolCall(
                "FINALIZE_CLASSIFICATION",
                {"final_evidence_state": "$state.final_evidence_state"},
            )
        if call.tool not in callable_skills:
            raise ActionError(
                f"fixed pipeline requires {call.tool}, but it is not callable from the current state"
            )
        raw = json.dumps(call.to_dict(), ensure_ascii=False, separators=(",", ":"))
        return PolicyDecision(call=call, raw_response=raw)


class ScriptedSkillExecutor:
    """Return predefined Observations while recording executed tools."""

    def __init__(self, outputs: Mapping[str, Sequence[Observation]]) -> None:
        self.outputs = {name: deque(values) for name, values in outputs.items()}
        self.executed_tools: list[str] = []

    def execute(
        self,
        call: ToolCall,
        record: MetadataRecord,
        state: Mapping[str, Any],
        contract: SkillContract,
    ) -> Observation:
        del record, state, contract
        self.executed_tools.append(call.tool)
        queue = self.outputs.get(call.tool)
        if not queue:
            return Observation(call.tool, "error", "no scripted output", {})
        return queue.popleft()


def _canonical_call(call: ToolCall) -> str:
    return json.dumps(call.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _validate_call_arguments(
    call: ToolCall,
    spec: SkillSpec,
    record: MetadataRecord,
    state: Mapping[str, Any],
) -> str | None:
    allowed = set(spec.required_inputs) | set(spec.optional_inputs)
    unknown = sorted(set(call.arguments) - allowed)
    if unknown:
        return f"unknown arguments for {call.tool}: {unknown}"
    for name in spec.required_inputs:
        if name not in call.arguments:
            return f"missing required argument {name!r}"
    if (call.tool == "INFER_CATEGORY_FROM_TOPK"
            and call.arguments.get("candidate_standard_categories")
            != "$state.standard_candidates"):
        return (
            "candidate_standard_categories must be the exact reference "
            "$state.standard_candidates; do not rewrite prior candidate IDs"
        )
    for name, value in call.arguments.items():
        found, _ = resolve_reference(value, record, state)
        if not found:
            return f"argument {name!r} cannot be resolved"
    return None


class AgentLoopController:
    """Run one validated Skill at a time with controller-owned safeguards."""

    def __init__(
        self,
        *,
        policy: LoopPolicy,
        executor: SkillExecutor,
        catalog: Mapping[str, SkillSpec] | None = None,
        skill_loader: SkillLoader,
        config: LoopConfig | None = None,
        registry_categories: Iterable[str] = (),
        registry_entries: Mapping[str, Mapping[str, Any]] | None = None,
        approved_levels: Iterable[str] = ("L1", "L2", "L3", "L4"),
        callable_derivation=None,
        state_publication=None,
    ) -> None:
        self.policy = policy
        self.executor = executor
        self.catalog = dict(catalog or dynamic_loop_catalog())
        self.skill_loader = skill_loader
        self.config = config or LoopConfig()
        self.registry_entries = {
            str(category): dict(entry)
            for category, entry in (registry_entries or {}).items()
        }
        self.registry_categories = frozenset(
            {str(item) for item in registry_categories} | set(self.registry_entries)
        )
        self.approved_levels = frozenset(str(item) for item in approved_levels)
        self.callable_derivation = callable_derivation or derive_callable_skills
        self.state_publication = state_publication

    def _normalize_policy_decision(self, decision, record, state, spec):
        """Extension point for controller-owned transport normalization."""
        return decision

    def _rejected_policy_metadata(self, decision, error):
        return {}

    def _valid_terminal(self, result: Any, state: Mapping[str, Any]) -> bool:
        if not isinstance(result, Mapping):
            return False
        category = result.get("category")
        level = result.get("grading_level")
        if not isinstance(category, str) or not category:
            return False
        if self.registry_categories and category not in self.registry_categories:
            return False
        if not isinstance(level, str) or level not in self.approved_levels:
            return False

        frozen_entry = self.registry_entries.get(category)
        if frozen_entry is not None and frozen_entry.get("grading_level") != level:
            return False

        aligned = state.get("aligned_standard_entry")
        if not isinstance(aligned, Mapping):
            return False
        aligned_category = aligned.get("category") or aligned.get("category_id")
        if aligned_category != category or aligned.get("grading_level") != level:
            return False

        candidates = state.get("standard_candidates")
        if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
            return False
        if not any(
            isinstance(candidate, Mapping)
            and (candidate.get("category") or candidate.get("category_id")) == category
            for candidate in candidates
        ):
            return False

        evidence = state.get("final_evidence_state")
        return (
            isinstance(evidence, Mapping)
            and evidence.get("category") == category
            and evidence.get("grading_level") == level
        )

    def _normalize_observation(
        self,
        call: ToolCall,
        observation: Observation,
        state: Mapping[str, Any],
    ) -> Observation:
        """Bind model-selected evidence back to exact frozen controller objects."""

        if observation.status != "success":
            return observation
        if call.tool in {"INFER_CATEGORY", "INFER_CATEGORY_FROM_TOPK"}:
            category = observation.data.get("category")
            candidates = state.get("standard_candidates")
            matched = None
            if isinstance(category, str) and isinstance(candidates, Sequence):
                matched = next(
                    (
                        candidate
                        for candidate in candidates
                        if isinstance(candidate, Mapping)
                        and (candidate.get("category") or candidate.get("category_id"))
                        == category
                    ),
                    None,
                )
            if not isinstance(matched, Mapping):
                return Observation(
                    call.tool,
                    "error",
                    "model-selected category is not one of the retrieved candidates",
                    {},
                    metadata=dict(observation.metadata),
                )
            data = dict(observation.data)
            original = data.get("aligned_standard_entry")
            if call.tool == "INFER_CATEGORY_FROM_TOPK":
                frozen = self.registry_entries.get(category)
                if not isinstance(frozen, Mapping):
                    return Observation(call.tool, "error", "selected category is absent from frozen registry", {})
                data["aligned_standard_entry"] = {"category": category, **dict(frozen)}
            else:
                data["aligned_standard_entry"] = dict(matched)
            metadata = dict(observation.metadata)
            metadata["controller_normalized_aligned_entry"] = original != data["aligned_standard_entry"]
            return Observation(
                observation.skill,
                observation.status,
                observation.summary,
                data,
                metadata=metadata,
            )
        if call.tool == "INFER_GRADING_LEVEL":
            level = observation.data.get("grading_level")
            if not isinstance(level, str) or level not in self.approved_levels:
                return Observation(
                    call.tool,
                    "error",
                    "model-produced grading level is empty or not approved",
                    {},
                    metadata=dict(observation.metadata),
                )
        return observation

    def _ready_finalization(self, state: Mapping[str, Any]) -> bool:
        """Require the current bundle to agree with exact frozen registry evidence."""
        evidence = state.get("final_evidence_state")
        if not self._valid_terminal(evidence, state):
            return False
        category = state.get("category")
        frozen = self.registry_entries.get(category) if isinstance(category, str) else None
        return (
            isinstance(frozen, Mapping)
            and evidence.get("category") == category
            and evidence.get("grading_level") == state.get("grading_level")
            and state.get("aligned_standard_entry") == {"category": category, **dict(frozen)}
            and evidence == self._final_evidence_state(None, state)
        )

    def _available_evidence(self, record, state):
        return _two_stage_evidence(record, state)

    def _final_evidence_state(self, record, state):
        return build_final_evidence_state(state)

    def run(self, record: MetadataRecord) -> LoopTrace:
        visible_record = MetadataRecord.from_mapping(
            {"domain": record.domain, "field_name": record.field_name,
             "source_id": record.source_id}
        )
        state = LoopState.from_record(visible_record, max_turns=self.config.max_turns)
        trace = LoopTrace(input=dict(state.input), stop_reason="running")
        identical_counts: Counter[str] = Counter()
        repetition_rejections = 0
        invalid_terminal_count = 0
        feedback: str | None = None
        recovery_attempted = False

        while state.turns < self.config.max_turns:
            forced = state.remaining_turns == 1
            recovering = False
            callable_skills = self.callable_derivation(visible_record, state, self.catalog)
            decision: PolicyDecision | None = None
            if forced:
                if "FINALIZE_CLASSIFICATION" not in callable_skills:
                    trace.events.append(
                        LoopEvent(
                            "controller_intervention",
                            state.turns + 1,
                            {"reason": "forced_finalization_missing_evidence"},
                        )
                    )
                    trace.stop_reason = "budget_exhausted_no_result"
                    break
                call = ToolCall(
                    "FINALIZE_CLASSIFICATION",
                    {"final_evidence_state": "$state.final_evidence_state"},
                )
                raw = json.dumps(call.to_dict(), ensure_ascii=False, separators=(",", ":"))
                decision = PolicyDecision(call=call, raw_response=raw)
                callable_skills = {
                    "FINALIZE_CLASSIFICATION": callable_skills["FINALIZE_CLASSIFICATION"]
                }
                trace.events.append(
                    LoopEvent(
                        "controller_intervention",
                        state.turns + 1,
                        {"reason": "forced_finalization", "call": call.to_dict()},
                    )
                )
            else:
                invalid_reason = "invalid_action"
                for attempt in range(self.config.action_retries + 1):
                    try:
                        decision = self.policy.select_action(
                            visible_record,
                            state,
                            callable_skills,
                            feedback=feedback,
                            forced_finalize=False,
                        )
                        if decision.call.tool not in callable_skills:
                            invalid_reason = "uncallable_skill"
                            raise ActionError(
                                f"{decision.call.tool} is not callable; choices={list(callable_skills)}"
                            )
                        spec = callable_skills[decision.call.tool]
                        decision = self._normalize_policy_decision(
                            decision, visible_record, state.named_state, spec
                        )
                        argument_error = _validate_call_arguments(
                            decision.call, spec, visible_record, state.named_state
                        )
                        if argument_error:
                            invalid_reason = "invalid_arguments"
                            raise ActionError(argument_error)
                        break
                    except ActionError as exc:
                        if isinstance(exc, EarlyStopError):
                            invalid_reason = "stop_without_final_result"
                        rejected_data: dict[str, Any] = {
                            "reason": invalid_reason,
                            "error": str(exc),
                            "attempt": attempt + 1,
                        }
                        if decision is not None:
                            rejected_data["raw_response"] = decision.raw_response
                            rejected_data["messages"] = [dict(m) for m in decision.messages]
                            rejected_data["metadata"] = dict(decision.metadata)
                        elif isinstance(getattr(exc, "metadata", None), Mapping):
                            rejected_data["metadata"] = dict(exc.metadata)
                        rejection_metadata = self._rejected_policy_metadata(decision, exc)
                        if rejection_metadata:
                            rejected_data.setdefault("metadata", {}).update(rejection_metadata)
                        decision = None
                        feedback = str(exc)
                        trace.events.append(
                            LoopEvent(
                                "retry" if attempt < self.config.action_retries else "rejected_action",
                                state.turns + 1,
                                rejected_data,
                            )
                        )
                        if attempt == self.config.action_retries:
                            if (
                                self.config.recover_ready_finalization
                                and not recovery_attempted
                                and "FINALIZE_CLASSIFICATION" in callable_skills
                                and self._ready_finalization(state.named_state)
                            ):
                                recovery_attempted = True
                                recovering = True
                                forced = True
                                call = ToolCall(
                                    "FINALIZE_CLASSIFICATION",
                                    {"final_evidence_state": "$state.final_evidence_state"},
                                )
                                decision = PolicyDecision(
                                    call=call,
                                    raw_response=json.dumps(call.to_dict(), ensure_ascii=False),
                                )
                                trace.events.append(LoopEvent(
                                    "controller_intervention", state.turns + 1,
                                    {"reason": "invalid_action_finalization_recovery", "call": call.to_dict()},
                                ))
                                break
                            if invalid_reason == "stop_without_final_result":
                                trace.stop_reason = "stop_without_final_result"
                            else:
                                trace.stop_reason = "invalid_action"
                            trace.turns = state.turns
                            trace.state = dict(state.named_state)
                            return trace

            assert decision is not None
            if forced:
                decision = self._normalize_policy_decision(
                    decision, visible_record, state.named_state, self.catalog[decision.call.tool]
                )
            call = decision.call
            canonical = _canonical_call(call)
            if identical_counts[canonical] >= self.config.max_identical_calls:
                repetition_rejections += 1
                recoverable = (
                    not recovering
                    and repetition_rejections <= self.config.repetition_retries
                )
                trace.events.append(
                    LoopEvent(
                        "controller_intervention",
                        state.turns + 1,
                        {
                            "reason": "repetition_limit",
                            "call": call.to_dict(),
                            "recovered": recoverable,
                            "attempt": repetition_rejections,
                            "messages": [dict(m) for m in decision.messages],
                            "raw_response": decision.raw_response,
                            "metadata": dict(decision.metadata),
                        },
                    )
                )
                if recoverable:
                    feedback = (
                        f"Repeated call {call.tool} was blocked; choose a different callable Skill."
                    )
                    continue
                trace.stop_reason = "repetition_limit"
                trace.turns = state.turns
                trace.state = dict(state.named_state)
                return trace
            identical_counts[canonical] += 1

            if decision.metadata.get("fixed_metadata_prefetch"):
                trace.events.append(
                    LoopEvent(
                        "controller_intervention", state.turns + 1,
                        {"reason": "metadata_prefetch", "call": call.to_dict()},
                    )
                )
            trace.events.append(
                LoopEvent(
                    "policy_action",
                    state.turns + 1,
                    {
                        "call": call.to_dict(),
                        "raw_response": decision.raw_response,
                        "messages": [dict(message) for message in decision.messages],
                        "metadata": dict(decision.metadata),
                        "forced_finalize": forced,
                    },
                )
            )
            contract = self.skill_loader.load(call.tool)
            try:
                observation = self.executor.execute(
                    call, visible_record, dict(state.named_state), contract
                )
            except Exception as exc:
                metadata = getattr(exc, "metadata", {})
                observation = Observation(
                    call.tool, "error", str(exc), {},
                    metadata=dict(metadata) if isinstance(metadata, Mapping) else {},
                )
            observation = self._normalize_observation(
                call, observation, state.named_state
            )

            state.turns += 1
            state.call_history.append(call.to_dict())
            trace.events.append(
                LoopEvent(
                    "observation",
                    state.turns,
                    {
                        "skill": observation.skill,
                        "status": observation.status,
                        "summary": observation.summary,
                        "data": dict(observation.data),
                        "metadata": dict(observation.metadata),
                    },
                )
            )
            feedback = None

            if observation.status != "success":
                feedback = f"Skill {call.tool} failed: {observation.summary}"
                if recovering:
                    trace.stop_reason = "recovery_finalization_failed"
                    break
                if forced or state.turns >= self.config.max_turns:
                    trace.stop_reason = "budget_exhausted_no_result"
                    break
                continue

            state.named_state.update(observation.data)
            if self.state_publication is not None:
                self.state_publication(visible_record, state.named_state, observation.data, call.tool)
            if call.tool == "GET_FIELD_METADATA" and "INTERPRET_FIELD_DESCRIPTION" in self.catalog:
                metadata = observation.data.get("field_metadata")
                if isinstance(metadata, Mapping):
                    for key in ("field_description", "table_description"):
                        value = metadata.get(key)
                        if isinstance(value, str) and value.strip():
                            state.named_state[key] = value
            if "INFER_CATEGORY_FROM_TOPK" in self.catalog:
                state.named_state["available_evidence"] = self._available_evidence(
                    visible_record, state.named_state
                )
            evidence = self._final_evidence_state(visible_record, state.named_state)
            if evidence is None:
                state.named_state.pop("final_evidence_state", None)
            else:
                state.named_state["final_evidence_state"] = evidence
            if call.tool != "FINALIZE_CLASSIFICATION":
                continue

            result = observation.data.get("final_result")
            if self._valid_terminal(result, state.named_state):
                trace.terminated = True
                trace.final_result = dict(result)
                trace.stop_reason = "completed"
                break

            invalid_terminal_count += 1
            feedback = "terminal result must contain one registry category and approved grading level"
            trace.events.append(
                LoopEvent(
                    "retry",
                    state.turns,
                    {"reason": "invalid_terminal", "result": result},
                )
            )
            if recovering:
                trace.stop_reason = "invalid_terminal"
                break
            if invalid_terminal_count > self.config.forced_finalize_retries:
                trace.stop_reason = "invalid_terminal"
                break
            if forced:
                trace.stop_reason = "budget_exhausted_no_result"
                break

        if trace.stop_reason == "running":
            trace.stop_reason = "budget_exhausted_no_result"
        trace.turns = state.turns
        trace.state = dict(state.named_state)
        return trace


__all__ = [
    "ActionError",
    "AgentLoopController",
    "EarlyStopError",
    "FixedPipelinePolicy",
    "LoopConfig",
    "LoopEvent",
    "LoopPolicy",
    "LoopState",
    "PrefetchNeighborFieldsPolicy",
    "PrefetchFieldMetadataPolicy",
    "LoopTrace",
    "POLICY_SYSTEM_PROMPT",
    "PolicyDecision",
    "ScriptedLoopPolicy",
    "ScriptedSkillExecutor",
    "SkillExecutor",
    "build_final_evidence_state",
    "build_policy_messages",
    "derive_callable_skills",
    "parse_policy_action",
]
