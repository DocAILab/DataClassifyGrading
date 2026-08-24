"""Testable direct-cascade rollout and RLOO contracts.

The production RL path is a *direct* cascade: Stage 1 proposes a candidate
bundle and every valid proposal (including a bundle which misses the target)
is sent to Stage 2.  Stage 2 therefore never receives an oracle target or a
fixture ``ground-truth + negatives`` bundle.  This module deliberately keeps
rollout and policy-framework details behind a tiny injected adapter so the
state machine can be exercised on CPU with a fake rollout.

The model-facing cascade contract is intentionally separate from the legacy
five-field RL parquet adapter:

* prompts expose ``field_name`` only; standards are represented by the
  registry/corpus/rubric assets and remain internal provenance;
* Stage 1 emits exactly ``K`` candidate choice ids;
* Stage 2 strictly emits ``{"answer": ..., "level": ...}`` (local candidate
  id plus approved level code); ``leaf`` and ``data_level`` are internal
  decoded names only;
* a trajectory owns both response segments and one advantage;
* RLOO baselines are computed by original source group, never by stage.

Stage 2 uses the shared ``build_stage2_prompt``/choice parser contract rather
than introducing a second model-facing schema.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import inspect
import json
from math import isfinite
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence, runtime_checkable

from agent.task.contracts import CorpusCategory, GradingConfig, LeafRegistry
from agent.task.parser import check_stage1_choices, check_stage2_choices
from agent.task.prompt_choices import PromptChoiceRegistry
from agent.task.prompts import Prompt, build_stage1_prompt, build_stage2_prompt


CASCADE_K = 5
CASCADE_N = 4
STAGE1_WEIGHT = 0.1
LEAF_WEIGHT = 0.6
LEVEL_WEIGHT = 0.3


@dataclass(frozen=True)
class CascadeConfig:
    """Fixed policy knobs for the formal cascade adapter.

    ``k`` and ``n`` are intentionally constrained to the accepted release
    policy.  Keeping them explicit in a dataclass makes accidental changes in
    a launcher or test immediately visible.
    """

    k: int = CASCADE_K
    n: int = CASCADE_N
    stage1_weight: float = STAGE1_WEIGHT
    leaf_weight: float = LEAF_WEIGHT
    level_weight: float = LEVEL_WEIGHT
    metadata_fields: tuple[str, ...] = ("field_name",)

    def __post_init__(self) -> None:
        if self.k != CASCADE_K:
            raise ValueError(f"cascade K is fixed at {CASCADE_K}")
        if self.n != CASCADE_N:
            raise ValueError(f"cascade N is fixed at {CASCADE_N}")
        if self.metadata_fields != ("field_name",):
            raise ValueError("cascade prompt fields are fixed to field_name")
        weights = (self.stage1_weight, self.leaf_weight, self.level_weight)
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in weights):
            raise ValueError("cascade reward weights must be numeric")
        if any(value < 0 or not isfinite(float(value)) for value in weights):
            raise ValueError("cascade reward weights must be finite and non-negative")
        if abs(sum(weights) - 1.0) > 1e-9:
            raise ValueError("cascade reward weights must sum to 1")


@dataclass(frozen=True)
class CascadeSource:
    """The prompt-visible and target metadata for one source record.

    ``ground_truth`` is used only for grading.  It is never passed to an
    adapter or used while constructing a Stage 2 candidate bundle.
    """

    source_id: str
    source_group: str
    field_name: str
    ground_truth: str
    ground_truth_level: str
    dataset: str = "unknown"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.source_id.strip():
            raise ValueError("cascade source_id must be non-empty")
        if not self.source_group.strip():
            raise ValueError("cascade source_group must be non-empty")
        if not isinstance(self.field_name, str) or not self.field_name.strip():
            raise ValueError("cascade field_name must be non-empty")
        if not self.ground_truth.strip():
            raise ValueError("cascade ground_truth must be non-empty")
        if not isinstance(self.ground_truth_level, str) or not self.ground_truth_level.strip():
            raise ValueError("cascade ground_truth_level must be non-empty")

    @classmethod
    def from_record(cls, record: Mapping[str, Any], *, index: int = 0) -> "CascadeSource":
        """Build a source from canonical or lightweight test-record mappings.

        Multiple spellings are accepted at this seam because upstream
        preprocessing versions used ``group_id`` and ``original_source_group``
        interchangeably.  Prompt fields are always normalized to strings.
        """

        if not isinstance(record, Mapping):
            raise ValueError(f"cascade source {index} must be a mapping")
        metadata = record.get("metadata")
        metadata_map = metadata if isinstance(metadata, Mapping) else {}

        def first(*keys: str, default: Any = "") -> Any:
            for key in keys:
                value = record.get(key)
                if value is not None and str(value).strip():
                    return value
                value = metadata_map.get(key)
                if value is not None and str(value).strip():
                    return value
            return default

        source_id = str(first("source_id", "id", default=f"source-{index}")).strip()
        source_group = str(
            first(
                "source_group",
                "original_source_group",
                "group_id",
                "source_id",
                "id",
                default=source_id,
            )
        ).strip()
        field_name = str(first("field_name", default="")).strip()
        dataset = str(first("dataset", default="unknown")).strip() or "unknown"

        target = record.get("target")
        target_map = target if isinstance(target, Mapping) else {}
        ground_truth = str(
            first("ground_truth", "category_id", default=target_map.get("category_id", ""))
            or ""
        ).strip()
        level = first(
            "ground_truth_level",
            "data_level",
            default="",
        )
        level_text = "" if level is None else str(level).strip()
        return cls(
            source_id=source_id,
            source_group=source_group,
            field_name=field_name,
            ground_truth=ground_truth,
            ground_truth_level=level_text or None,
            dataset=dataset,
            metadata=dict(metadata_map),
        )


@dataclass(frozen=True)
class RolloutResponse:
    """Adapter response with optional policy-token provenance.

    A fake adapter may return a plain string; :func:`normalize_rollout` turns
    it into this structure.  ``response_mask`` defaults to one for every
    token, while an explicit mask allows framework adapters to exclude prompt
    or padding tokens from the policy loss.
    """

    text: str
    token_ids: tuple[int, ...] = ()
    logprobs: tuple[float, ...] = ()
    response_mask: tuple[int, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise ValueError("rollout response text must be a string")
        if self.response_mask and len(self.response_mask) != len(self.token_ids):
            raise ValueError("response_mask must parallel token_ids")
        if self.logprobs and len(self.logprobs) != len(self.token_ids):
            raise ValueError("logprobs must parallel token_ids")
        if any(mask not in (0, 1, False, True) for mask in self.response_mask):
            raise ValueError("response_mask entries must be 0/1")

    @property
    def mask(self) -> tuple[int, ...]:
        return self.response_mask or tuple(1 for _ in self.token_ids)


# Friendly alias used by adapter implementations.
RolloutResult = RolloutResponse


def normalize_rollout(value: Any) -> RolloutResponse:
    """Normalize a fake/framework rollout result without importing a backend."""

    if isinstance(value, RolloutResponse):
        return value
    if isinstance(value, str):
        return RolloutResponse(value)
    if isinstance(value, Mapping):
        text = value.get("text", value.get("response", value.get("output", "")))
        token_ids = value.get("token_ids", value.get("tokens", ()))
        logprobs = value.get("logprobs", value.get("token_logprobs", ()))
        response_mask = value.get("response_mask", value.get("loss_mask", ()))
        return RolloutResponse(
            str(text),
            tuple(int(token) for token in token_ids or ()),
            tuple(float(prob) for prob in logprobs or ()),
            tuple(int(mask) for mask in response_mask or ()),
            {str(key): item for key, item in value.items() if key not in {
                "text", "response", "output", "token_ids", "tokens",
                "logprobs", "token_logprobs", "response_mask", "loss_mask",
            }},
        )
    text = getattr(value, "text", getattr(value, "response", getattr(value, "output", None)))
    if text is not None:
        return RolloutResponse(
            str(text),
            tuple(int(token) for token in (getattr(value, "token_ids", ()) or ())),
            tuple(float(prob) for prob in (getattr(value, "logprobs", ()) or ())),
            tuple(int(mask) for mask in (getattr(value, "response_mask", ()) or ())),
        )
    raise TypeError("rollout adapter must return text or a response mapping/object")


@dataclass(frozen=True)
class ResponseSegment:
    """One policy-trained response segment in a trajectory."""

    stage: str
    response: RolloutResponse
    valid: bool
    loss_mask: tuple[int, ...]
    advantage: float | None = None

    def __post_init__(self) -> None:
        if self.stage not in {"stage1", "stage2"}:
            raise ValueError("response segment stage must be stage1 or stage2")
        if len(self.loss_mask) != len(self.response.token_ids):
            raise ValueError("segment loss_mask must parallel response token_ids")

    @property
    def text(self) -> str:
        return self.response.text

    @property
    def token_ids(self) -> tuple[int, ...]:
        return self.response.token_ids

    @property
    def logprobs(self) -> tuple[float, ...]:
        return self.response.logprobs

    def with_advantage(self, advantage: float) -> "ResponseSegment":
        return replace(self, advantage=float(advantage))


@dataclass(frozen=True)
class Stage1Outcome:
    response: RolloutResponse
    valid: bool
    candidates: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    hit: bool = False


@dataclass(frozen=True)
class Stage2Outcome:
    response: RolloutResponse
    valid: bool
    leaf: str | None = None
    data_level: str | None = None
    errors: tuple[str, ...] = ()

    @property
    def answer(self) -> str | None:
        return self.leaf

    @property
    def level(self) -> str | None:
        return self.data_level


@dataclass(frozen=True)
class CascadeTrajectory:
    """Complete direct-cascade lineage for one Stage 1 sibling."""

    source_id: str
    source_group: str
    ground_truth: str
    ground_truth_level: str | None
    stage1: Stage1Outcome
    stage2: Stage2Outcome | None
    stage2_candidates: tuple[str, ...]
    stage2_prompt: Prompt | None
    reward: float
    stage1_hit: bool
    leaf_correct: bool
    level_correct: bool
    terminated: bool
    termination_reason: str | None
    segments: tuple[ResponseSegment, ResponseSegment]
    advantage: float | None = None

    def __post_init__(self) -> None:
        if len(self.segments) != 2 or tuple(segment.stage for segment in self.segments) != (
            "stage1", "stage2"
        ):
            raise ValueError("cascade trajectories must carry stage1 and stage2 segments")
        if self.reward < 0 or self.reward > 1 or not isfinite(float(self.reward)):
            raise ValueError("cascade reward must be finite and within [0, 1]")
        if self.terminated and self.stage2 is not None and self.stage2.valid:
            raise ValueError("a terminated trajectory cannot have a valid stage2 outcome")

    @property
    def group_id(self) -> str:
        # RLOO siblings are the N complete trajectories sampled for one
        # original field record. Table/domain grouping must never mix
        # baselines across distinct source ids.
        return self.source_id

    @property
    def candidate_lineage(self) -> tuple[str, ...]:
        """Actual Stage 1 candidates used to construct Stage 2."""

        return self.stage2_candidates

    @property
    def candidates(self) -> tuple[str, ...]:
        return self.stage2_candidates

    @property
    def stage1_outcome(self) -> Stage1Outcome:
        return self.stage1

    @property
    def stage2_outcome(self) -> Stage2Outcome | None:
        return self.stage2

    @property
    def responses(self) -> tuple[ResponseSegment, ResponseSegment]:
        return self.segments

    def with_advantage(self, advantage: float) -> "CascadeTrajectory":
        value = float(advantage)
        return replace(
            self,
            advantage=value,
            segments=tuple(segment.with_advantage(value) for segment in self.segments),
        )


@dataclass(frozen=True)
class CascadeRun:
    trajectories: tuple[CascadeTrajectory, ...]

    @property
    def rewards(self) -> tuple[float, ...]:
        return tuple(trajectory.reward for trajectory in self.trajectories)

    @property
    def advantages(self) -> tuple[float, ...]:
        return tuple(
            float(trajectory.advantage) if trajectory.advantage is not None else 0.0
            for trajectory in self.trajectories
        )

    def by_group(self) -> dict[str, tuple[CascadeTrajectory, ...]]:
        grouped: dict[str, list[CascadeTrajectory]] = {}
        for trajectory in self.trajectories:
            grouped.setdefault(trajectory.source_id, []).append(trajectory)
        return {key: tuple(value) for key, value in grouped.items()}


@runtime_checkable
class RolloutAdapter(Protocol):
    """Minimal injectable adapter expected by :class:`CascadeRunner`.

    Implementations may return strings or :class:`RolloutResponse` objects.
    The runner also recognizes ``stage1``/``stage2`` method names for tiny
    test doubles.
    """

    def rollout_stage1(
        self, prompt: Prompt, *, n: int, source: CascadeSource
    ) -> Sequence[Any]: ...

    def rollout_stage2(
        self,
        prompt: Prompt,
        *,
        candidates: Sequence[str],
        source: CascadeSource,
    ) -> Any: ...


def _metadata_for_source(source: CascadeSource) -> dict[str, str]:
    # ``standards`` is intentionally not a free-form prompt field.  The
    # canonical registry/corpus/rubric assets carry the standard context; this
    # helper exposes only the accepted ``field_name`` metadata contract.
    return {"field_name": source.field_name}


def build_cascade_stage1_prompt(
    source: CascadeSource,
    registry: LeafRegistry,
) -> Prompt:
    """Build the fixed field_name+standards Stage 1 prompt."""

    # Importing TaskConfig lazily keeps this module's public contract small and
    # avoids introducing any algorithm state into ``agent.task``.
    from agent.task.contracts import TaskConfig

    return build_stage1_prompt(
        _metadata_for_source(source),
        registry,
        TaskConfig(("field_name",)),
    )


def _corpus_entry(corpus: Mapping[str, CorpusCategory], category_id: str) -> CorpusCategory:
    try:
        entry = corpus[category_id]
    except KeyError as exc:
        raise ValueError(
            f"Stage 1 candidate {category_id!r} is absent from the canonical corpus"
        ) from exc
    if not isinstance(entry, CorpusCategory):
        raise ValueError(f"corpus entry {category_id!r} is not a CorpusCategory")
    return entry


def build_cascade_stage2_prompt(
    source: CascadeSource,
    candidates: Sequence[str],
    registry: LeafRegistry,
    corpus: Mapping[str, CorpusCategory],
    *,
    grading: GradingConfig | None = None,
) -> Prompt:
    """Build Stage 2 from the actual decoded Stage 1 candidate lineage.

    The shared prompt builder owns the strict model-facing ``answer``/``level``
    schema and local-id mapping.  Passing the candidate tuple here (rather
    than rebuilding it from a target) is the no-oracle guarantee.
    """

    if len(candidates) != CASCADE_K or len(set(candidates)) != CASCADE_K:
        raise ValueError(f"cascade Stage 2 requires exactly {CASCADE_K} unique candidates")
    if any(candidate not in registry.ids for candidate in candidates):
        raise ValueError("cascade Stage 2 candidates must belong to the registry")
    if not corpus:
        raise ValueError("cascade Stage 2 requires a canonical corpus")
    for category_id in candidates:
        _corpus_entry(corpus, category_id)
    from agent.task.contracts import TaskConfig

    return build_stage2_prompt(
        _metadata_for_source(source),
        candidates,
        registry,
        TaskConfig(("field_name",)),
        corpus=corpus,
        grading=grading,
    )


def decode_stage1_candidates(
    text: str,
    *,
    registry: LeafRegistry,
    choices: PromptChoiceRegistry | None = None,
    expected_count: int = CASCADE_K,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Strictly decode a Stage 1 response through the shared choice parser."""

    if expected_count != CASCADE_K:
        raise ValueError(f"cascade K is fixed at {CASCADE_K}")
    choices = choices or PromptChoiceRegistry.from_registry(registry)
    result = check_stage1_choices(
        text,
        choices=choices,
        expected_count=expected_count,
    )
    if not result.format_valid or not result.constraint_valid:
        return (), tuple(result.errors)
    assert isinstance(result.decoded, tuple)
    return tuple(result.decoded), ()


def decode_stage2_output(
    text: str,
    *,
    candidates: Sequence[str],
    grading: GradingConfig | None = None,
) -> tuple[str | None, str | None, tuple[str, ...]]:
    """Strictly decode the shared ``{"answer", "level"}`` Stage 2 shape.

    ``leaf`` and ``data_level`` are internal names used by trajectory records;
    they are deliberately not accepted as model output.  Local answer ids are
    mapped only through the actual candidate list supplied by Stage 1.
    """

    if len(candidates) != CASCADE_K or len(set(candidates)) != CASCADE_K:
        raise ValueError("stage2 candidates must contain exactly five unique ids")
    if grading is None:
        raise ValueError("formal cascade Stage 2 requires a grading config")
    result = check_stage2_choices(text, candidates=candidates, grading=grading)
    if not result.format_valid or not result.constraint_valid:
        return None, None, tuple(result.errors)
    assert isinstance(result.decoded, str)
    return result.decoded, result.level, ()


def cascade_reward(
    *,
    stage1_hit: bool,
    leaf_correct: bool,
    level_correct: bool,
    valid: bool,
    config: CascadeConfig | None = None,
) -> float:
    """Compute the fixed reward; every invalid trajectory receives zero."""

    config = config or CascadeConfig()
    if not valid:
        return 0.0
    return float(
        config.stage1_weight * bool(stage1_hit)
        + config.leaf_weight * bool(leaf_correct)
        + config.level_weight * bool(level_correct)
    )


def leave_one_out_advantages(
    rewards: Sequence[float] | Mapping[str, Sequence[float]],
    groups: Sequence[str] | Mapping[str, Sequence[float]] | None = None,
) -> tuple[float, ...] | dict[str, tuple[float, ...]]:
    """Return exact leave-one-out advantages grouped by source group.

    For each sibling ``i`` in group ``g``:

    ``A_i = r_i - (sum(r_g) - r_i) / (|g| - 1)``.

    There is no cross-group baseline and no stage-level duplication.  Mapping
    input is supported for callers that already grouped rewards and preserves
    each group's insertion order.
    """

    if groups is None:
        if not isinstance(rewards, Mapping):
            raise ValueError("groups are required when rewards are a flat sequence")
        groups = rewards
    if isinstance(groups, Mapping):
        result: dict[str, tuple[float, ...]] = {}
        for group, values in groups.items():
            result[str(group)] = _loo_group(tuple(float(value) for value in values), str(group))
        return result
    if isinstance(rewards, Mapping) or len(rewards) != len(groups):
        raise ValueError("rewards and groups must have the same length")
    grouped_indices: dict[str, list[int]] = {}
    for index, group in enumerate(groups):
        grouped_indices.setdefault(str(group), []).append(index)
    output = [0.0] * len(rewards)
    for group, indices in grouped_indices.items():
        values = tuple(float(rewards[index]) for index in indices)
        advantages = _loo_group(values, group)
        for index, advantage in zip(indices, advantages):
            output[index] = advantage
    return tuple(output)


def _loo_group(values: tuple[float, ...], group: str) -> tuple[float, ...]:
    if len(values) < 2:
        raise ValueError(f"RLOO group {group!r} must contain at least two rollouts")
    if any(not isfinite(value) for value in values):
        raise ValueError(f"RLOO group {group!r} contains a non-finite reward")
    total = sum(values)
    denominator = len(values) - 1
    return tuple(value - (total - value) / denominator for value in values)


# Common naming in training code and tests.
compute_rloo_advantages = leave_one_out_advantages
rloo_advantages = leave_one_out_advantages


def _call_with_supported_kwargs(function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Call adapters with a documented signature while tolerating tiny fakes."""

    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return function(*args, **kwargs)
    accepts_var_kw = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if accepts_var_kw:
        return function(*args, **kwargs)
    allowed = {
        name: value for name, value in kwargs.items() if name in signature.parameters
    }
    return function(*args, **allowed)


def _stage1_call(
    adapter: Any,
    prompt: Prompt,
    source: CascadeSource,
    n: int,
) -> Sequence[Any]:
    function = getattr(adapter, "rollout_stage1", None) or getattr(adapter, "stage1", None)
    if function is None and callable(adapter):
        function = adapter
    if function is None:
        raise TypeError("rollout adapter must implement rollout_stage1 or stage1")
    result = _call_with_supported_kwargs(function, prompt, n=n, source=source)
    if isinstance(result, (str, bytes, Mapping, RolloutResponse)):
        # A single mapping/string is accepted only when N=1; fixed policy N=4
        # should always return siblings, so fail loudly rather than duplicate.
        raise ValueError("Stage 1 adapter must return exactly N rollout responses")
    values = tuple(result)
    if len(values) != n:
        raise ValueError(f"Stage 1 adapter returned {len(values)} responses; expected {n}")
    return values


def _stage2_call(
    adapter: Any,
    prompt: Prompt,
    source: CascadeSource,
    candidates: Sequence[str],
) -> Any:
    function = getattr(adapter, "rollout_stage2", None) or getattr(adapter, "stage2", None)
    if function is None:
        raise TypeError("rollout adapter must implement rollout_stage2 or stage2")
    return _call_with_supported_kwargs(
        function,
        prompt,
        candidates=candidates,
        source=source,
    )


class CascadeRunner:
    """Execute direct-cascade rollouts and attach source-group RLOO values."""

    def __init__(
        self,
        *,
        registry: LeafRegistry,
        corpus: Mapping[str, CorpusCategory],
        rollout: RolloutAdapter | Any,
        grading: GradingConfig | None = None,
        config: CascadeConfig | None = None,
    ) -> None:
        self.registry = registry
        self.corpus = corpus
        self.rollout = rollout
        if not isinstance(grading, GradingConfig):
            raise ValueError("formal cascade requires an approved grading config")
        self.grading = grading
        self.config = config or CascadeConfig()
        if not corpus:
            raise ValueError("cascade requires a canonical corpus")
        if any(category_id not in corpus for category_id in registry.ids):
            missing = sorted(set(registry.ids) - set(corpus))
            raise ValueError(f"cascade corpus is missing registry categories: {missing}")

    def run(self, records: Iterable[Mapping[str, Any] | CascadeSource]) -> CascadeRun:
        # Accept either a flat record iterable or a pre-grouped mapping used by
        # small CPU tests: {original_source_group: [record, ...]}.
        if isinstance(records, Mapping) and not any(
            key in records for key in ("id", "source_id", "target", "ground_truth")
        ):
            flattened: list[Mapping[str, Any] | CascadeSource] = []
            for group, group_records in records.items():
                if isinstance(group_records, (Mapping, CascadeSource)):
                    group_records = (group_records,)
                for record in group_records:
                    if isinstance(record, CascadeSource):
                        flattened.append(record)
                    elif isinstance(record, Mapping):
                        flattened.append({**record, "source_group": str(group)})
                    else:
                        raise ValueError("pre-grouped cascade records must be mappings")
            records = flattened
        sources = tuple(
            record if isinstance(record, CascadeSource) else CascadeSource.from_record(record, index=index)
            for index, record in enumerate(records)
        )
        if not sources:
            raise ValueError("cascade requires at least one source record")
        trajectories: list[CascadeTrajectory] = []
        for source in sources:
            stage1_prompt = build_cascade_stage1_prompt(source, self.registry)
            responses = _stage1_call(self.rollout, stage1_prompt, source, self.config.n)
            for raw_response in responses:
                stage1_response = normalize_rollout(raw_response)
                candidates, errors = decode_stage1_candidates(
                    stage1_response.text,
                    registry=self.registry,
                    expected_count=self.config.k,
                )
                stage1_valid = not errors
                stage1_outcome = Stage1Outcome(
                    response=stage1_response,
                    valid=stage1_valid,
                    candidates=candidates,
                    errors=errors,
                    hit=stage1_valid and source.ground_truth in candidates,
                )
                stage1_segment = ResponseSegment(
                    "stage1",
                    stage1_response,
                    stage1_valid,
                    stage1_response.mask,
                )
                if not stage1_valid:
                    stage2_response = RolloutResponse("")
                    stage2_outcome = None
                    stage2_segment = ResponseSegment("stage2", stage2_response, False, ())
                    trajectories.append(
                        CascadeTrajectory(
                            source_id=source.source_id,
                            source_group=source.source_id,
                            ground_truth=source.ground_truth,
                            ground_truth_level=source.ground_truth_level,
                            stage1=stage1_outcome,
                            stage2=stage2_outcome,
                            stage2_candidates=(),
                            stage2_prompt=None,
                            reward=0.0,
                            stage1_hit=False,
                            leaf_correct=False,
                            level_correct=False,
                            terminated=True,
                            termination_reason="invalid_stage1",
                            segments=(stage1_segment, stage2_segment),
                        )
                    )
                    continue

                # Crucially, this list is exactly the decoded Stage 1 output:
                # no target injection, no fixture negatives, no oracle path.
                stage2_prompt = build_cascade_stage2_prompt(
                    source,
                    candidates,
                    self.registry,
                    self.corpus,
                    grading=self.grading,
                )
                raw_stage2 = _stage2_call(self.rollout, stage2_prompt, source, candidates)
                stage2_response = normalize_rollout(raw_stage2)
                leaf, level, stage2_errors = decode_stage2_output(
                    stage2_response.text,
                    candidates=candidates,
                    grading=self.grading,
                )
                stage2_valid = not stage2_errors
                stage2_outcome = Stage2Outcome(
                    response=stage2_response,
                    valid=stage2_valid,
                    leaf=leaf,
                    data_level=level,
                    errors=stage2_errors,
                )
                stage1_hit = source.ground_truth in candidates
                leaf_correct = stage2_valid and leaf == source.ground_truth
                level_correct = (
                    stage2_valid
                    and source.ground_truth_level is not None
                    and level == source.ground_truth_level
                )
                reward = cascade_reward(
                    stage1_hit=stage1_hit,
                    leaf_correct=leaf_correct,
                    level_correct=level_correct,
                    valid=stage2_valid,
                    config=self.config,
                )
                stage2_segment = ResponseSegment(
                    "stage2",
                    stage2_response,
                    stage2_valid,
                    stage2_response.mask,
                )
                trajectories.append(
                    CascadeTrajectory(
                        source_id=source.source_id,
                        source_group=source.source_id,
                        ground_truth=source.ground_truth,
                        ground_truth_level=source.ground_truth_level,
                        stage1=stage1_outcome,
                        stage2=stage2_outcome,
                        stage2_candidates=candidates,
                        stage2_prompt=stage2_prompt,
                        reward=reward,
                        stage1_hit=stage1_hit,
                        leaf_correct=leaf_correct,
                        level_correct=level_correct,
                        terminated=False,
                        termination_reason=None,
                        segments=(stage1_segment, stage2_segment),
                    )
                )
        groups = tuple(trajectory.source_id for trajectory in trajectories)
        advantages = leave_one_out_advantages(
            tuple(trajectory.reward for trajectory in trajectories), groups
        )
        return CascadeRun(
            tuple(trajectory.with_advantage(advantage) for trajectory, advantage in zip(trajectories, advantages))
        )


# Alias spelling used by some launchers.
CascadeRolloutRunner = CascadeRunner


def trajectory_to_policy_records(run: CascadeRun) -> tuple[dict[str, Any], ...]:
    """Flatten both response segments while retaining one trajectory advantage.

    Stage 1 and Stage 2 rows intentionally share ``advantage``.  A terminated
    Stage 1 trajectory still emits a masked Stage 2 placeholder, making it
    impossible for a batching layer to accidentally train only one segment.
    """

    records: list[dict[str, Any]] = []
    for trajectory in run.trajectories:
        for segment in trajectory.segments:
            records.append(
                {
                    "source_id": trajectory.source_id,
                    "source_group": trajectory.source_id,
                    "stage": segment.stage,
                    "response": segment.text,
                    "token_ids": list(segment.token_ids),
                    "logprobs": list(segment.logprobs),
                    "loss_mask": list(segment.loss_mask),
                    "advantage": trajectory.advantage,
                    "reward": trajectory.reward,
                    "terminated": trajectory.terminated,
                }
            )
    return tuple(records)


__all__ = [
    "CASCADE_K",
    "CASCADE_N",
    "STAGE1_WEIGHT",
    "LEAF_WEIGHT",
    "LEVEL_WEIGHT",
    "CascadeConfig",
    "CascadeSource",
    "RolloutResponse",
    "RolloutResult",
    "RolloutAdapter",
    "ResponseSegment",
    "Stage1Outcome",
    "Stage2Outcome",
    "CascadeTrajectory",
    "CascadeRun",
    "CascadeRunner",
    "CascadeRolloutRunner",
    "build_cascade_stage1_prompt",
    "build_cascade_stage2_prompt",
    "decode_stage1_candidates",
    "decode_stage2_output",
    "cascade_reward",
    "leave_one_out_advantages",
    "compute_rloo_advantages",
    "rloo_advantages",
    "trajectory_to_policy_records",
    "normalize_rollout",
]
