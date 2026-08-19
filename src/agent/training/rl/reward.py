"""Shared task reward for the two-stage classification task.

The reward describes TASK correctness only — it contains no training
algorithm logic (no advantage, no KL, no normalization). It consumes the
unified parser (check_stage1_output / check_stage2_output) so malformed
model output can never raise inside a training loop; every failure mode
maps to a structured RewardResult instead.

Reward table (Stage 1):
    format/schema/category invalid ......... 0.0
    valid but ground truth not in candidates  config.stage1_valid_miss (0.3)
    valid and ground truth in candidates .... 1.0

Reward table (Stage 2):
    format/schema/answer invalid ............ 0.0
    valid answer but wrong .................. config.stage2_partial
    correct .................................. 1.0

Stage 2 partial value is PROVISIONAL / pending task-policy confirmation;
``stage2_partial`` stays a configuration item and must not be read as a
final reward policy.

Stage 2 semantics: the ground truth is NOT required to be among the
candidates — a Stage 1 recall failure (ground_truth not in candidates) is a
legal task state, not a Stage 2 contract error. A valid candidate answer
still computes a reward without raising.

Choice protocol: reward_stage1_choices / reward_stage2_choices /
reward_for_choice_result consume the SHARED choice-aware parser
(check_stage1_choices / check_stage2_choices), which decodes the model's
choice ids to canonical category ids BEFORE the SAME reward table below
applies. Choice validation, partial-credit values and malformed-output
handling are therefore exactly the canonical ones; RewardConfig and the
reward numbers are unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

from agent.task.contracts import LeafRegistry
from agent.task.parser import (
    ChoiceParseResult,
    ParseResult,
    Stage1Output,
    Stage2Output,
    check_stage1_choices,
    check_stage1_output,
    check_stage2_choices,
    check_stage2_output,
)
from agent.task.prompt_choices import PromptChoiceRegistry

FULL_REWARD = 1.0
INVALID_REWARD = 0.0
STAGE1_VALID_MISS_DEFAULT = 0.3
STAGE2_PARTIAL_DEFAULT = 0.5


@dataclass(frozen=True)
class RewardConfig:
    """Reward weights.

    ``stage2_partial`` is a configuration item on purpose: its value is
    PROVISIONAL / PENDING task-policy confirmation and will be finalized
    with the first real RL vertical slice. No value tuning happens before
    that confirmation.
    """

    stage1_valid_miss: float = STAGE1_VALID_MISS_DEFAULT
    stage2_partial: float = STAGE2_PARTIAL_DEFAULT

    def __post_init__(self) -> None:
        for name, value in (
            ("stage1_valid_miss", self.stage1_valid_miss),
            ("stage2_partial", self.stage2_partial),
        ):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"reward config {name} must be a number")
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"reward config {name} must be within [0, 1]")


@dataclass(frozen=True)
class RewardResult:
    """Structured outcome of one task reward computation.

    - reward: the scalar task reward.
    - reason: human-readable justification (safe to log).
    - parsed_output: the parsed model output when format-valid (for choice
      outputs: the decoded canonical shape when decodable, else the raw
      model-level shape).
    - format_valid: whether the output was valid JSON with the exact schema.
    - constraint_valid: whether the output satisfied the task constraints
      (Stage 1 candidate rules / Stage 2 answer membership) — distinguishes
      "valid but wrong" (partial credit) from "invalid" (zero).
    - task_correct: whether the output achieved the task goal (Stage 1:
      ground truth recalled; Stage 2: correct answer).
    """

    reward: float
    reason: str
    parsed_output: Stage1Output | Stage2Output | None
    format_valid: bool
    constraint_valid: bool
    task_correct: bool


def _invalid(reason: str) -> RewardResult:
    return RewardResult(INVALID_REWARD, reason, None, False, False, False)


def reward_for_parse_result(
    stage: str,
    result: ParseResult,
    *,
    ground_truth: str,
    config: RewardConfig | None = None,
) -> RewardResult:
    """Map a unified ParseResult to a task reward without re-parsing.

    Single implementation of the reward tables; reward_stage1 / reward_stage2
    are thin wrappers that run the unified parser first.
    """
    config = config or RewardConfig()
    if stage == "stage1":
        if not result.format_valid:
            return _invalid("stage1 output is not a valid JSON object with schema candidates")
        if not result.constraint_valid:
            return RewardResult(
                INVALID_REWARD,
                "stage1 output violates candidate constraints: " + "; ".join(result.errors),
                result.output,
                True,
                False,
                False,
            )
        assert isinstance(result.output, Stage1Output)
        if ground_truth in result.output.candidates:
            return RewardResult(FULL_REWARD, "stage1 candidates contain the ground truth", result.output, True, True, True)
        return RewardResult(
            config.stage1_valid_miss,
            "stage1 output is valid but misses the ground truth",
            result.output,
            True,
            True,
            False,
        )
    if stage == "stage2":
        if not result.format_valid:
            return _invalid("stage2 output is not a valid JSON object with schema answer")
        if not result.constraint_valid:
            return RewardResult(
                INVALID_REWARD,
                "stage2 answer is not one of the candidates",
                result.output,
                True,
                False,
                False,
            )
        assert isinstance(result.output, Stage2Output)
        if result.output.answer == ground_truth:
            return RewardResult(FULL_REWARD, "stage2 answer is correct", result.output, True, True, True)
        return RewardResult(
            config.stage2_partial,
            "stage2 answer is valid but wrong",
            result.output,
            True,
            True,
            False,
        )
    raise ValueError(f"reward stage must be stage1 or stage2, got {stage!r}")


def reward_stage1(
    solution: str,
    *,
    ground_truth: str,
    registry: LeafRegistry,
    config: RewardConfig | None = None,
) -> RewardResult:
    """Stage 1 task reward: recall of the ground truth among 5 candidates.

    Never raises on model output; ``ground_truth`` outside the registry is a
    programming error and raises (mirrors evaluate_stage1).
    """
    if ground_truth not in registry.ids:
        raise ValueError("ground_truth must belong to the leaf registry")
    result = check_stage1_output(solution, registry=registry)
    return reward_for_parse_result("stage1", result, ground_truth=ground_truth, config=config)


def reward_stage2(
    solution: str,
    *,
    ground_truth: str,
    candidates: tuple[str, ...] | list[str],
    registry: LeafRegistry,
    config: RewardConfig | None = None,
) -> RewardResult:
    """Stage 2 task reward: correct answer among the candidate bundle.

    Never raises on model output. ``candidates`` must be 5 unique registry
    IDs (programming contract, mirrors evaluate_stage2).
    """
    if ground_truth not in registry.ids:
        raise ValueError("ground_truth must belong to the leaf registry")
    if isinstance(candidates, (str, bytes)) or (
        len(candidates) != 5
        or len(set(candidates)) != 5
        or any(candidate not in registry.ids for candidate in candidates)
    ):
        raise ValueError("candidates must be 5 unique IDs from the leaf registry")
    result = check_stage2_output(solution, candidates=candidates)
    return reward_for_parse_result("stage2", result, ground_truth=ground_truth, config=config)


def reward_for_choice_result(
    stage: str,
    result: ChoiceParseResult,
    *,
    ground_truth: str,
    config: RewardConfig | None = None,
) -> RewardResult:
    """Map a shared choice-aware ``ChoiceParseResult`` to a task reward.

    Single implementation of the reward tables: the decoded canonical view
    feeds ``reward_for_parse_result``, so choice validation, partial-credit
    values and malformed-output handling are exactly the canonical ones.
    Never raises on model output.
    """
    return reward_for_parse_result(
        stage, result.canonical_view(), ground_truth=ground_truth, config=config
    )


def reward_stage1_choices(
    solution: str,
    *,
    ground_truth: str,
    registry: LeafRegistry,
    choices: PromptChoiceRegistry | None = None,
    config: RewardConfig | None = None,
) -> RewardResult:
    """Stage 1 task reward over the choice protocol (global choice ids).

    The model answers with choice ids; the shared choice-aware parser
    (check_stage1_choices) decodes them to canonical category ids BEFORE
    the unchanged reward table applies. Never raises on model output;
    ``ground_truth`` outside the registry is a programming error and
    raises (mirrors reward_stage1).
    """
    if ground_truth not in registry.ids:
        raise ValueError("ground_truth must belong to the leaf registry")
    choices = choices or PromptChoiceRegistry.from_registry(registry)
    result = check_stage1_choices(solution, choices=choices)
    return reward_for_choice_result(
        "stage1", result, ground_truth=ground_truth, config=config
    )


def reward_stage2_choices(
    solution: str,
    *,
    ground_truth: str,
    candidates: tuple[str, ...] | list[str],
    registry: LeafRegistry,
    config: RewardConfig | None = None,
) -> RewardResult:
    """Stage 2 task reward over the choice protocol (local bundle id 1..5).

    The model answers with a local id; the shared choice-aware parser
    (check_stage2_choices) decodes it against the candidate bundle BEFORE
    the unchanged reward table applies. Never raises on model output;
    ``candidates`` must be 5 unique registry IDs (programming contract,
    mirrors reward_stage2).
    """
    if ground_truth not in registry.ids:
        raise ValueError("ground_truth must belong to the leaf registry")
    if isinstance(candidates, (str, bytes)) or (
        len(candidates) != 5
        or len(set(candidates)) != 5
        or any(candidate not in registry.ids for candidate in candidates)
    ):
        raise ValueError("candidates must be 5 unique IDs from the leaf registry")
    result = check_stage2_choices(solution, candidates=candidates)
    return reward_for_choice_result(
        "stage2", result, ground_truth=ground_truth, config=config
    )


__all__ = [
    "FULL_REWARD",
    "INVALID_REWARD",
    "STAGE1_VALID_MISS_DEFAULT",
    "STAGE2_PARTIAL_DEFAULT",
    "RewardConfig",
    "RewardResult",
    "reward_stage1",
    "reward_stage2",
    "reward_for_parse_result",
    "reward_stage1_choices",
    "reward_stage2_choices",
    "reward_for_choice_result",
]
