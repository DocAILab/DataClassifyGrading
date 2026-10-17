"""Model-independent adapters for policy generation and semantic Skill execution."""

from __future__ import annotations

import json
from dataclasses import replace
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .agent_loop import (
    ActionError,
    LoopState,
    PolicyDecision,
    _two_stage_evidence,
    build_policy_messages,
    parse_policy_action,
)
from .catalog import SkillSpec
from .contracts import OPTIONAL_METADATA_FIELDS, MetadataRecord, ToolCall
from .metadata_lookup import FieldMetadataLookup
from .table_field_lookup import TableFieldLookup
from .category_catalog import CategoryCatalog
from .runtime import KnowledgeBase, Observation
from .skill_loader import SkillContract, SkillLoader
from .validator import resolve_reference
from .topk_selection import SELECTION_POLICY, aggregate_rankings, balanced_options, ranking_scores


TextGenerator = Callable[[list[dict[str, str]]], str]
DeterministicHandler = Callable[[ToolCall, MetadataRecord, Mapping[str, Any]], Observation]


def _generation_metadata(generate: TextGenerator) -> dict[str, Any]:
    metadata = getattr(generate, "last_generation_metadata", None)
    return dict(metadata) if isinstance(metadata, Mapping) else {}


def _attempt_metadata(attempts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Keep every Skill generation while preserving the existing aggregate fields."""

    last = attempts[-1]
    metadata: dict[str, Any] = {
        "attempts": [dict(item) for item in attempts],
        "model_call_count": len(attempts),
        "messages": last["messages"],
        "raw_response": last.get("raw_response"),
    }
    generations = [item.get("generation", {}) for item in attempts]
    usages = [item.get("token_usage") for item in generations if isinstance(item, Mapping)]
    if any(isinstance(item, Mapping) for item in usages):
        metadata["token_usage"] = {
            key: sum(int(item.get(key) or 0) for item in usages if isinstance(item, Mapping))
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        }
    elapsed = [item.get("elapsed_seconds") for item in generations if isinstance(item, Mapping)]
    if any(isinstance(item, (int, float)) for item in elapsed):
        metadata["elapsed_seconds"] = sum(
            float(item) for item in elapsed if isinstance(item, (int, float))
        )
    return metadata


class SkillExecutionError(ValueError):
    """Raised when a Skill cannot produce one valid Observation."""

    def __init__(self, message: str, *, metadata: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.metadata = dict(metadata or {})


def metadata_availability_handler(
    call: ToolCall,
    record: MetadataRecord,
    resolved: Mapping[str, Any],
) -> Observation:
    """Report availability from the projected, model-visible record only."""

    del resolved
    available = [name for name in ("domain", "field_name") if record.available(name)]
    missing = [name for name in OPTIONAL_METADATA_FIELDS if not record.available(name)]
    return Observation(
        call.tool,
        "success",
        f"Available metadata: {', '.join(available)}; missing: {', '.join(missing)}",
        {"metadata_availability": {"available": available, "missing": missing}},
    )


def make_standard_search_handler(
    knowledge_base: KnowledgeBase,
    *,
    top_k: int = 5,
) -> DeterministicHandler:
    """Create a frozen-registry Top-K retrieval handler."""

    if top_k < 1:
        raise ValueError("top_k must be positive")

    def handler(
        call: ToolCall,
        record: MetadataRecord,
        resolved: Mapping[str, Any],
    ) -> Observation:
        del record
        query = resolved.get("semantic_query")
        candidates = knowledge_base.search_standard(query, top_k=top_k)
        return Observation(
            call.tool,
            "success",
            f"Retrieved {len(candidates)} standard candidates",
            {"standard_candidates": candidates},
        )

    return handler


def make_field_metadata_handler(lookup: FieldMetadataLookup) -> DeterministicHandler:
    """Expose only allowlisted source metadata when the Agent invokes this Skill."""

    def handler(
        call: ToolCall,
        record: MetadataRecord,
        resolved: Mapping[str, Any],
    ) -> Observation:
        del resolved
        fields = lookup.get(record)
        return Observation(
            call.tool,
            "success",
            f"Retrieved {len(fields)} approved metadata fields",
            {"field_metadata": fields},
        )

    return handler


def make_table_fields_handler(lookup: TableFieldLookup) -> DeterministicHandler:
    """Expose bounded, label-free same-table fields only on an explicit call."""

    def handler(
        call: ToolCall,
        record: MetadataRecord,
        resolved: Mapping[str, Any],
    ) -> Observation:
        del resolved
        fields = lookup.get(record)
        return Observation(
            call.tool,
            "success",
            f"Retrieved {len(fields['fields'])} of {fields['available_count']} neighboring fields",
            {"table_fields": fields},
        )

    return handler


def _decode_observation_object(text: str) -> Mapping[str, Any]:
    if not isinstance(text, str) or not text.strip():
        raise SkillExecutionError("Skill response must be non-empty JSON")
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines.pop()
        cleaned = "\n".join(lines).strip()
    if not cleaned.startswith("{"):
        raise SkillExecutionError("Skill response must be one JSON object")
    decoder = json.JSONDecoder()
    try:
        value, end = decoder.raw_decode(cleaned)
    except JSONDecodeError as exc:
        raise SkillExecutionError(f"Skill response is invalid JSON: {exc.msg}") from exc
    if cleaned[end:].strip():
        raise SkillExecutionError("Skill response contains trailing content")
    if not isinstance(value, Mapping):
        raise SkillExecutionError("Skill response must be a JSON object")
    return value


def parse_skill_observation(text: str, spec: SkillSpec) -> Observation:
    """Parse one strict success Observation for the selected Skill."""

    payload = _decode_observation_object(text)
    if set(payload) != {"summary", "data"}:
        raise SkillExecutionError("Skill response must contain exactly summary and data")
    summary = payload.get("summary")
    data = payload.get("data")
    if not isinstance(summary, str) or not summary.strip():
        raise SkillExecutionError("Skill summary must be a non-empty string")
    if not isinstance(data, Mapping):
        raise SkillExecutionError("Skill data must be an object")
    expected = set(spec.produces)
    actual = set(data)
    missing = sorted(expected - actual)
    if missing:
        raise SkillExecutionError(f"missing produced keys: {missing}")
    extra = sorted(actual - expected)
    if extra:
        raise SkillExecutionError(f"unexpected produced keys: {extra}")
    empty = sorted(
        key
        for key in expected
        if data.get(key) is None or data.get(key) == ""
    )
    if empty:
        raise SkillExecutionError(f"produced keys must be non-empty: {empty}")
    return Observation(spec.name, "success", summary.strip(), dict(data))


def load_few_shot_messages(path: str | Path) -> tuple[dict[str, str], ...]:
    """Load synthetic policy examples while rejecting evaluation-label leakage."""

    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("few-shot file must contain a non-empty JSON list")
    messages: list[dict[str, str]] = []
    forbidden = ("ground_truth", "ground truth", "source_id", "source id")
    for index, item in enumerate(payload):
        if not isinstance(item, Mapping) or set(item) != {"role", "content"}:
            raise ValueError(f"few-shot message {index} must contain exactly role and content")
        role = item.get("role")
        content = item.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str) or not content:
            raise ValueError(f"few-shot message {index} has an invalid role or content")
        lowered = content.lower()
        if any(marker in lowered for marker in forbidden):
            raise ValueError("few-shot examples must not contain ground-truth or source identifiers")
        messages.append({"role": role, "content": content})
    return tuple(messages)


class ModelLoopPolicy:
    """Adapt a text generator to the one-action dynamic policy contract."""

    def __init__(
        self,
        generate: TextGenerator,
        *,
        few_shot_messages: Sequence[Mapping[str, str]] = (),
        metadata_guidance: bool = False,
        two_stage: bool = False,
        description_skills: bool = False,
    ) -> None:
        self.generate = generate
        self.few_shot_messages = tuple(dict(message) for message in few_shot_messages)
        self.metadata_guidance = metadata_guidance
        self.two_stage = two_stage
        self.description_skills = description_skills

    def select_action(
        self,
        record: MetadataRecord,
        state: LoopState,
        callable_skills: Mapping[str, SkillSpec],
        *,
        feedback: str | None = None,
        forced_finalize: bool = False,
    ) -> PolicyDecision:
        messages = build_policy_messages(
            record,
            state,
            callable_skills,
            feedback=feedback,
            forced_finalize=forced_finalize,
        )
        if self.two_stage:
            messages[0]["content"] += (
                "\nUse only a Skill listed in callable_skills for this turn; "
                "never call SELECT_CATEGORY_TOPK before it is callable. "
                "If semantic evidence is absent, first call INTERPRET_FIELD_NAME "
                "with field_name=$metadata.field_name or another callable semantic Skill. "
                "When SELECT_CATEGORY_TOPK becomes callable, provide semantic_query="
                "$state.field_semantics if present, otherwise a different existing "
                "$state.<semantic_name>; never omit required arguments. "
                "Then call INFER_CATEGORY_FROM_TOPK with candidate_standard_categories="
                "$state.standard_candidates and all_available_evidence="
                "$state.available_evidence; bare $state is invalid. "
                "Only grade after selecting one category."
            )
            if self.metadata_guidance:
                messages[0]["content"] += (
                    " If GET_FIELD_METADATA is callable and field_metadata is absent, "
                    "call it before category selection."
                )
                if "table_fields" not in state.named_state:
                    if self.description_skills:
                        messages[0]["content"] += (
                            " Once real field_metadata is available, if "
                            "INTERPRET_FIELD_DESCRIPTION is callable and "
                            "field_description_semantics is absent, call "
                            "INTERPRET_FIELD_DESCRIPTION before SELECT_CATEGORY_TOPK "
                            "with field_description=$state.field_description. "
                            "If no field description exists but "
                            "INTERPRET_TABLE_DESCRIPTION is callable, use its real "
                            "table_description instead. After interpretation, pass "
                            "semantic_query=$state.available_evidence to "
                            "SELECT_CATEGORY_TOPK and pass "
                            "all_available_evidence=$state.available_evidence to Top-1. "
                            "If neither description is available, use "
                            "semantic_query=$state.field_metadata."
                        )
                    else:
                        messages[0]["content"] += (
                            " Once field_metadata is available, pass semantic_query="
                            "$state.field_metadata to SELECT_CATEGORY_TOPK and pass "
                            "$state.field_metadata to INFER_CATEGORY_FROM_TOPK."
                        )
            if "GET_TABLE_FIELDS" in callable_skills and "table_fields" not in state.named_state:
                messages[0]["content"] += (
                    " If the current field and table metadata leave the business "
                    "category ambiguous, you may call GET_TABLE_FIELDS with "
                    "current_record=$record before category selection; this is optional."
                )
            if "table_fields" in state.named_state:
                messages[0]["content"] += (
                    " Neighboring table_fields are available. To use both target "
                    "metadata and neighbors in category selection, pass semantic_query="
                    "$state.available_evidence to SELECT_CATEGORY_TOPK and "
                    "all_available_evidence=$state.available_evidence to "
                    "INFER_CATEGORY_FROM_TOPK."
                )
            if self.metadata_guidance:
                messages[0]["content"] += (
                    " Finish grading and finalization without repeating diagnostic Skills."
                )
            if self.description_skills and "field_metadata" in state.named_state:
                messages[0]["content"] += (
                    " Real descriptions in field_metadata may be interpreted through "
                    "INTERPRET_FIELD_DESCRIPTION or INTERPRET_TABLE_DESCRIPTION only "
                    "when these Skills appear in callable_skills. They are optional: "
                    "do not call a missing-description Skill or repeat one. "
                    "If description semantics were produced, use "
                    "semantic_query=$state.available_evidence for SELECT_CATEGORY_TOPK "
                    "and all_available_evidence=$state.available_evidence for Top-1."
                )
        elif self.metadata_guidance:
            messages[0]["content"] += (
                    "\nIf GET_FIELD_METADATA is callable and field_metadata is absent, "
                    "call GET_FIELD_METADATA before semantic interpretation or retrieval. "
                    "When real field_metadata has been obtained and standard_candidates "
                    "are absent, call SEARCH_STANDARD with "
                    "{\"semantic_query\":\"$state.field_metadata\"}; do not search only "
                    "the guessed field_semantics. Once candidates are available, "
                    "proceed toward category, level, and finalization without "
                    "repeating diagnostic Skills."
            )
        if self.few_shot_messages:
            messages[1:1] = [dict(message) for message in self.few_shot_messages]
        try:
            raw = self.generate(messages)
        except Exception as exc:  # pragma: no cover - backend-specific
            error = ActionError(f"policy generator failed: {exc}")
            error.metadata = {"messages": messages, "raw_response": None,
                              "error": str(exc), "generation": _generation_metadata(self.generate)}
            raise error from exc
        if not isinstance(raw, str):
            raise ActionError("policy generator must return a string")
        try:
            parsed = parse_policy_action(raw)
        except ActionError as exc:
            exc.metadata = {"messages": messages, "raw_response": raw,
                            "generation": _generation_metadata(self.generate)}
            raise
        return PolicyDecision(
            call=parsed,
            raw_response=raw,
            messages=tuple(messages),
            metadata=_generation_metadata(self.generate),
        )


class ModelSkillExecutor:
    """Execute deterministic tools or one selected Markdown Skill via a generator."""

    def __init__(
        self,
        generate: TextGenerator,
        loader: SkillLoader,
        *,
        deterministic_handlers: Mapping[str, DeterministicHandler] | None = None,
        category_catalog: CategoryCatalog | None = None,
        top_k: int = 5,
        guard_two_stage_evidence: bool = False,
        topk_evidence_mode: str = "augmented",
        topk_selection_mode: str = "single",
        topk_selection_seed: int = 20260930,
        top1_evidence_mode: str = "all",
        top1_selection_mode: str = "model",
        evidence_assembly=None,
    ) -> None:
        if top1_evidence_mode not in {"all", "raw"}:
            raise ValueError("top1_evidence_mode must be all or raw")
        if top1_selection_mode not in {"model", "first"}:
            raise ValueError("top1_selection_mode must be model or first")
        self.top1_evidence_mode = top1_evidence_mode
        self.top1_selection_mode = top1_selection_mode
        if topk_evidence_mode not in {"augmented", "raw", "raw-flat"}:
            raise ValueError("topk_evidence_mode must be augmented, raw or raw-flat")
        if topk_selection_mode not in {"single", "balanced-vote"}:
            raise ValueError("topk_selection_mode must be single or balanced-vote")
        if topk_selection_mode == "balanced-vote" and category_catalog is None:
            raise ValueError("balanced-vote requires a two-stage category catalog")
        self.generate = generate
        self.loader = loader
        self.category_catalog = category_catalog
        self.top_k = top_k
        self.guard_two_stage_evidence = guard_two_stage_evidence
        self.evidence_assembly = evidence_assembly
        self.topk_evidence_mode = topk_evidence_mode
        self.topk_selection_mode = topk_selection_mode
        self.topk_selection_seed = topk_selection_seed
        self.deterministic_handlers = {
            "CHECK_METADATA_AVAILABILITY": metadata_availability_handler,
            **dict(deterministic_handlers or {}),
        }

    @staticmethod
    def _resolved_arguments(
        call: ToolCall,
        record: MetadataRecord,
        state: Mapping[str, Any],
    ) -> dict[str, Any]:
        resolved: dict[str, Any] = {}
        for name, value in call.arguments.items():
            found, result = resolve_reference(value, record, state)
            if not found:
                raise SkillExecutionError(f"argument {name!r} cannot be resolved")
            resolved[name] = result
        return resolved

    @staticmethod
    def _messages(contract: SkillContract, resolved: Mapping[str, Any]) -> list[dict[str, str]]:
        system = (
            f"Execute only the selected Skill {contract.name}.\n"
            f"DSL: {contract.dsl}\n"
            f"Contract:\n{contract.body}\n"
            "Use only the resolved inputs. Return exactly one JSON object with summary and data."
        )
        payload = {
            "skill": contract.name,
            "resolved_inputs": dict(resolved),
            "output_schema": {
                "summary": "non-empty natural-language observation",
                "data": {name: "required output" for name in contract.produces},
            },
        }
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
        ]

    @staticmethod
    def _validate_deterministic(observation: Observation, spec: SkillSpec) -> Observation:
        if observation.skill != spec.name:
            raise SkillExecutionError(
                f"deterministic handler returned Skill {observation.skill!r}, expected {spec.name!r}"
            )
        if observation.status != "success":
            return observation
        actual = set(observation.data)
        expected = set(spec.produces)
        if actual != expected:
            raise SkillExecutionError(
                f"deterministic handler output keys {sorted(actual)} do not match {sorted(expected)}"
            )
        return observation

    def execute(
        self,
        call: ToolCall,
        record: MetadataRecord,
        state: Mapping[str, Any],
        contract: SkillContract,
    ) -> Observation:
        loaded = self.loader.load(call.tool)
        if loaded != contract:
            raise SkillExecutionError(f"contract mismatch for {call.tool}")
        spec = self.loader.catalog[call.tool]
        resolved = self._resolved_arguments(call, record, state)
        if call.tool in {"SELECT_CATEGORY_TOPK", "INFER_CATEGORY_FROM_TOPK"}:
            try:
                return self._execute_two_stage(call, record, resolved, state, contract)
            except SkillExecutionError as exc:
                if call.tool == "INFER_CATEGORY_FROM_TOPK":
                    exc.metadata.update(top1_selection_mode=self.top1_selection_mode,
                                        top1_evidence_mode=self.top1_evidence_mode)
                raise
        if call.tool == "FINALIZE_CLASSIFICATION" and self.guard_two_stage_evidence:
            return self._finalize_guarded_two_stage(call, resolved, state)
        handler = self.deterministic_handlers.get(call.tool)
        if handler is not None:
            return self._validate_deterministic(handler(call, record, resolved), spec)

        messages = self._messages(contract, resolved)
        try:
            raw = self.generate(messages)
        except Exception as exc:  # pragma: no cover - backend-specific
            raise SkillExecutionError(f"Skill generator failed: {exc}", metadata={
                "messages": messages, "raw_response": None, "error": str(exc),
                "generation": _generation_metadata(self.generate)}) from exc
        if not isinstance(raw, str):
            raise SkillExecutionError("Skill generator must return a string")
        try:
            observation = parse_skill_observation(raw, spec)
        except SkillExecutionError as exc:
            exc.metadata = {"messages": messages, "raw_response": raw,
                            "generation": _generation_metadata(self.generate)}
            raise
        return replace(
            observation,
            metadata={
                "messages": messages,
                "raw_response": raw,
                **_generation_metadata(self.generate),
            },
        )

    def _finalize_guarded_two_stage(
        self,
        call: ToolCall,
        resolved: Mapping[str, Any],
        state: Mapping[str, Any],
    ) -> Observation:
        if self.category_catalog is None:
            raise SkillExecutionError("two-stage category catalog is not configured")
        evidence = resolved.get("final_evidence_state")
        if not isinstance(evidence, Mapping) or evidence != state.get("final_evidence_state"):
            raise SkillExecutionError("finalizer must consume the frozen final evidence state")
        category = evidence.get("category")
        level = evidence.get("grading_level")
        aligned = evidence.get("aligned_standard_entry")
        candidates = evidence.get("standard_candidates")
        if not isinstance(category, str) or not isinstance(level, str):
            raise SkillExecutionError("final evidence is missing category or grading level")
        if category != state.get("category") or level != state.get("grading_level"):
            raise SkillExecutionError("final evidence differs from the selected category or level")
        if (
            not isinstance(aligned, Mapping)
            or (aligned.get("category") or aligned.get("category_id")) != category
            or aligned.get("grading_level") != level
        ):
            raise SkillExecutionError("final evidence disagrees with the aligned entry")
        if not isinstance(candidates, list) or not any(
            isinstance(item, Mapping)
            and (item.get("category") or item.get("category_id")) == category
            for item in candidates
        ):
            raise SkillExecutionError("selected category is absent from frozen Top-K")
        try:
            registry_entry = self.category_catalog.frozen_entry(category)
        except ValueError as exc:
            raise SkillExecutionError(str(exc)) from exc
        if registry_entry.get("grading_level") != level:
            raise SkillExecutionError("selected level disagrees with frozen Registry")
        return Observation(
            call.tool,
            "success",
            "Bound final result to the selected category and frozen grading level",
            {"final_result": {"category": category, "grading_level": level}},
            metadata={"controller_bound_terminal": True},
        )

    def _execute_two_stage(
        self,
        call: ToolCall,
        record: MetadataRecord,
        resolved: Mapping[str, Any],
        state: Mapping[str, Any],
        contract: SkillContract,
    ) -> Observation:
        if self.category_catalog is None:
            raise SkillExecutionError("two-stage category catalog is not configured")
        evidence_assembled = (
            self.guard_two_stage_evidence
            and call.tool == "SELECT_CATEGORY_TOPK"
            and isinstance(state.get("field_metadata"), Mapping)
        )
        if call.tool == "SELECT_CATEGORY_TOPK":
            if evidence_assembled and self.evidence_assembly is None:
                semantic_query = _two_stage_evidence(record, state)
                if self.topk_evidence_mode == "raw":
                    semantic_query["observations"] = {
                        "field_metadata": state["field_metadata"],
                    }
                elif self.topk_evidence_mode == "raw-flat":
                    semantic_query = {
                        **state["field_metadata"],
                        "domain": record.domain, "field_name": record.field_name,
                    }
            else:
                semantic_query = resolved["semantic_query"]
            prompt_input = {
                "record_input": {"domain": record.domain, "field_name": record.field_name},
                "semantic_query": semantic_query,
                "top_k": self.top_k,
                "category_options": self.category_catalog.selection_options(),
            }
            if self.evidence_assembly is not None:
                # Opt-in pool callers preserve the selected native query and share the
                # same validated publication/final evidence boundary.
                prompt_input["available_evidence"] = self.evidence_assembly(record, state)
            output_rule = (
                'Return exactly one JSON object: {"option_ids":["C001",...],'
                f'"summary":"reason"}}. The list MUST contain exactly {self.top_k} distinct '
                f'option IDs from category_options, ranked best first. If fewer than {self.top_k} '
                'are strong matches, include weaker but plausible supplied options to fill every slot. '
                'Copy the short option_id exactly. Do not output long category paths, canonical '
                'category IDs, duplicates, invented option IDs, or a different top-N. '
                'Limit summary to one short phrase (at most 20 characters); do not put '
                'quotation marks, line breaks, or a long explanation inside summary. '
                'Close the summary string and the JSON object.'
            )
        else:
            prior = state.get("standard_candidates")
            supplied = resolved.get("candidate_standard_categories")
            if not isinstance(prior, list) or supplied != prior:
                raise SkillExecutionError("Top-1 must consume the actual prior Top-K observation")
            if len(prior) != self.top_k or any(not isinstance(item, Mapping) for item in prior):
                raise SkillExecutionError("Top-1 requires exactly top_k candidate objects")
            ids = [item.get("category") for item in prior if isinstance(item, Mapping)]
            try:
                self.category_catalog.validate_top_k(ids, self.top_k)
                details = self.category_catalog.option_details(ids)
            except ValueError as exc:
                raise SkillExecutionError(str(exc)) from exc
            if self.top1_selection_mode == "first":
                category = ids[0]
                return Observation(call.tool, "success", "Selected actual first frozen candidate",
                    {"category": category, "aligned_standard_entry": self.category_catalog.frozen_entry(category)},
                    metadata={"top1_selection_mode": "first", "top1_evidence_mode": self.top1_evidence_mode,
                              "controller_bound_selection": True, "model_call_count": 0, "attempts": []})
            evidence = resolved["all_available_evidence"]
            if self.top1_evidence_mode == "raw":
                evidence = {"domain": record.domain, "field_name": record.field_name,
                    "observations": {"field_metadata": dict(state["field_metadata"])}
                    if isinstance(state.get("field_metadata"), Mapping) else {}}
            prompt_input = {
                "all_available_evidence": evidence,
                "candidate_details": details,
            }
            if self.evidence_assembly is not None:
                prompt_input["available_evidence"] = self.evidence_assembly(record, state)
            output_rule = (
                'Return exactly one JSON object: {"option_id":"one supplied short option ID",'
                '"summary":"reason"}. Do not choose outside candidate_details. '
                'Limit summary to one short phrase (at most 20 characters); do not put '
                'quotation marks, line breaks, or a long explanation inside summary. '
                'Close the summary string and the JSON object.'
            )
        messages = [
            {"role": "system", "content": f"Execute only the selected Skill {contract.name}.\n{contract.body}\n{output_rule}"},
            {"role": "user", "content": json.dumps(prompt_input, ensure_ascii=False, indent=2)},
        ]
        if call.tool == "SELECT_CATEGORY_TOPK" and self.topk_selection_mode == "balanced-vote":
            return self._execute_balanced_topk(record, prompt_input, messages[0], evidence_assembled)
        attempts: list[dict[str, Any]] = []
        max_attempts = 2
        for attempt_index in range(max_attempts):
            try:
                raw = self.generate(messages)
            except Exception as exc:
                attempts.append({"messages": messages, "raw_response": None, "error": str(exc),
                                 "generation": _generation_metadata(self.generate)})
                raise SkillExecutionError(
                    f"two-stage Skill generator failed: {exc}",
                    metadata=_attempt_metadata(attempts),
                ) from exc
            generation = _generation_metadata(self.generate)
            try:
                payload = _decode_observation_object(raw)
                summary = payload.get("summary")
                if not isinstance(summary, str) or not summary.strip():
                    raise SkillExecutionError("two-stage Skill requires a nonempty summary")
                if call.tool == "SELECT_CATEGORY_TOPK":
                    if set(payload) != {"option_ids", "summary"}:
                        raise SkillExecutionError("Top-K response must contain option_ids and summary only")
                    canonical_ids = self.category_catalog.resolve_option_ids(
                        payload["option_ids"], self.top_k
                    )
                    label_map = {item["category"]: item for item in self.category_catalog.labels()}
                    data = {"standard_candidates": [label_map[item] for item in canonical_ids]}
                else:
                    if set(payload) != {"option_id", "summary"}:
                        raise SkillExecutionError("Top-1 response must contain option_id and summary only")
                    category = self.category_catalog.resolve_option_ids([payload["option_id"]], 1)[0]
                    if category not in ids:
                        raise SkillExecutionError("Top-1 category is outside prior Top-K")
                    data = {"category": category, "aligned_standard_entry": self.category_catalog.frozen_entry(category)}
            except (SkillExecutionError, ValueError, TypeError, KeyError) as exc:
                error = str(exc)
                attempts.append({"messages": messages, "raw_response": raw,
                                 "error": error, "generation": generation})
                if attempt_index + 1 < max_attempts:
                    if call.tool == "SELECT_CATEGORY_TOPK":
                        feedback = (
                            f"Your Top-K response failed validation: {error}. "
                            f"Return a complete new JSON object with exactly {self.top_k} "
                            "distinct option IDs copied from category_options. "
                            "Use option_ids and summary only; never use canonical paths or IDs. "
                            "Keep summary to one short phrase and close the JSON object."
                        )
                    else:
                        feedback = (
                            f"Your Top-1 response failed validation: {error}. "
                            "Return one complete JSON object with option_id copied from "
                            "candidate_details and a brief summary. Close the object with "
                            "a closing brace }. Keep summary to one short phrase. "
                            "Do not use any option outside candidate_details."
                        )
                    messages = [
                        *messages,
                        {"role": "assistant", "content": raw if isinstance(raw, str) else ""},
                        {"role": "user", "content": feedback},
                    ]
                    continue
                raise SkillExecutionError(
                    f"invalid two-stage Skill response: {error}",
                    metadata=_attempt_metadata(attempts),
                ) from exc
            attempts.append({"messages": messages, "raw_response": raw,
                             "generation": generation})
            metadata = _attempt_metadata(attempts)
            if call.tool == "INFER_CATEGORY_FROM_TOPK":
                metadata.update({"top1_evidence_mode": self.top1_evidence_mode,
                                 "top1_selection_mode": self.top1_selection_mode})
            if evidence_assembled:
                metadata["controller_evidence_assembled"] = True
                metadata["topk_evidence_mode"] = self.topk_evidence_mode
            return Observation(call.tool, "success", summary.strip(), data,
                               metadata=metadata)
        raise AssertionError("unreachable")

    def _execute_balanced_topk(
        self, record: MetadataRecord, prompt_input: Mapping[str, Any],
        system_message: Mapping[str, str], evidence_assembled: bool,
    ) -> Observation:
        """Make exactly three independent calls, retaining failures without retry."""
        assert self.category_catalog is not None
        attempts: list[dict[str, Any]] = []
        rankings: list[list[str]] = []
        options = prompt_input["category_options"]
        allowed = [item["option_id"] for item in options]
        for index, presented in enumerate(balanced_options(
            options, record.domain, record.field_name, self.topk_selection_seed
        ), start=1):
            messages = [dict(system_message), {
                "role": "user", "content": json.dumps(
                    {**prompt_input, "category_options": presented}, ensure_ascii=False, indent=2
                ),
            }]
            attempt: dict[str, Any] = {
                "round": index, "presented_option_ids": [item["option_id"] for item in presented],
                "valid": False, "error": None, "raw_response": None,
                "messages": messages, "generation": {},
            }
            # Most generators keep a last-call attribute. Clear it before this call
            # so a failing backend cannot be charged for the previous completion.
            metadata_cleared = False
            try:
                if hasattr(self.generate, "last_generation_metadata"):
                    self.generate.last_generation_metadata = {}
                metadata_cleared = True
            except (AttributeError, TypeError):
                pass
            try:
                raw = self.generate(messages)
            except Exception as exc:
                attempt["error"] = f"generator failed: {exc}"
                if metadata_cleared:
                    attempt["generation"] = _generation_metadata(self.generate)
                attempts.append(attempt)
                continue
            attempt["raw_response"] = raw
            attempt["generation"] = _generation_metadata(self.generate)
            try:
                payload = _decode_observation_object(raw)
                if set(payload) != {"option_ids", "summary"}:
                    raise SkillExecutionError("Top-K response must contain option_ids and summary only")
                if not isinstance(payload["summary"], str) or not payload["summary"].strip():
                    raise SkillExecutionError("two-stage Skill requires a nonempty summary")
                self.category_catalog.resolve_option_ids(payload["option_ids"], self.top_k)
                ranking = list(payload["option_ids"])
                rankings.append(ranking)
                attempt.update({"valid": True, "option_ids": ranking, "votes": [
                    {"option_id": item, "vote": 1, "reciprocal_rank_score": f"1/{rank}"}
                    for rank, item in enumerate(ranking, start=1)
                ]})
            except (SkillExecutionError, ValueError, TypeError, KeyError) as exc:
                attempt["error"] = str(exc)
            attempts.append(attempt)
        metadata = _attempt_metadata(attempts)
        metadata["format_retry_count"] = 0
        selection = {
            "mode": "balanced-vote", "policy": SELECTION_POLICY,
            "seed": self.topk_selection_seed, "rounds": 3, "quorum": 2,
            "valid_rounds": len(rankings), "selected_option_ids": [], "scores": [],
        }
        metadata["topk_selection"] = selection
        if evidence_assembled:
            metadata.update({"controller_evidence_assembled": True, "topk_evidence_mode": self.topk_evidence_mode})
        if len(rankings) < 2:
            raise SkillExecutionError("balanced-vote requires at least two valid rankings", metadata=metadata)
        selected = aggregate_rankings(rankings, allowed, record.domain, record.field_name,
                                      self.topk_selection_seed, self.top_k)
        selection["selected_option_ids"] = selected
        selection["scores"] = ranking_scores(rankings, allowed, record.domain, record.field_name,
                                             self.topk_selection_seed, self.top_k)
        canonical = self.category_catalog.resolve_option_ids(selected, self.top_k)
        labels = {item["category"]: item for item in self.category_catalog.labels()}
        return Observation("SELECT_CATEGORY_TOPK", "success", "Balanced Top-K vote",
                           {"standard_candidates": [labels[item] for item in canonical]}, metadata=metadata)


__all__ = [
    "DeterministicHandler",
    "ModelLoopPolicy",
    "ModelSkillExecutor",
    "SkillExecutionError",
    "TextGenerator",
    "make_standard_search_handler",
    "make_field_metadata_handler",
    "make_table_fields_handler",
    "metadata_availability_handler",
    "load_few_shot_messages",
    "parse_skill_observation",
]
