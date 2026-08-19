"""Algorithm-independent evaluation for two-stage leaf classification.

Canonical semantics: the final comparison object is ALWAYS the canonical
category_id (registry / corpus / ground truth). The LLM boundary speaks
choice ids instead:

    raw model output -> parse choice_id -> decode category_id
    -> existing canonical correctness logic (evaluate_stage1/2)

evaluate_stage1/2 consume the unified parser (check_stage1_output /
check_stage2_output) so the reward adapter and the evaluator share one
contract implementation. evaluate_stage1_choices / evaluate_stage2_choices
consume the SHARED choice-aware layer (check_stage1_choices /
check_stage2_choices) and apply the same canonical correctness facts on
the decoded category ids, so choice validation is implemented exactly
once across evaluation and reward.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from agent.task import LeafRegistry
from agent.task.parser import (
    Stage2Output,
    check_stage1_choices,
    check_stage1_output,
    check_stage2_choices,
    check_stage2_output,
)
from agent.task.prompt_choices import PromptChoiceRegistry


@dataclass(frozen=True)
class Stage1Evaluation:
    prediction: tuple[str, ...] | None
    format_valid: bool
    contract_valid: bool
    ground_truth_recalled: bool
    errors: tuple[str, ...]


@dataclass(frozen=True)
class Stage2Evaluation:
    prediction: str | None
    format_valid: bool
    contract_valid: bool
    correct: bool
    errors: tuple[str, ...]


def evaluate_stage1(
    solution: str,
    *,
    ground_truth: str,
    registry: LeafRegistry,
) -> Stage1Evaluation:
    """Evaluate format, candidate constraints, and Recall@5 separately.

    Correctness facts come from the unified parser (check_stage1_output) so
    the reward adapter and the evaluator share one contract implementation.
    """
    if ground_truth not in registry.ids:
        raise ValueError("ground_truth must belong to the leaf registry")
    result = check_stage1_output(solution, registry=registry)
    recalled = result.ok and ground_truth in result.output.candidates
    return Stage1Evaluation(
        result.output.candidates if result.format_valid else None,
        result.format_valid,
        result.ok,
        recalled,
        result.errors,
    )


def evaluate_stage1_choices(
    solution: str,
    *,
    ground_truth: str,
    registry: LeafRegistry,
    choices: PromptChoiceRegistry | None = None,
) -> Stage1Evaluation:
    """Evaluate a choice-id Stage 1 output; decode BEFORE canonical logic.

    Consumes the shared choice-aware check (check_stage1_choices) so
    evaluation and reward share one choice validation implementation; the
    returned prediction is the decoded canonical category_id tuple. Invalid
    choice ids / wrong counts / duplicates yield an explicit invalid result
    with no name or fuzzy fallback.
    """
    if ground_truth not in registry.ids:
        raise ValueError("ground_truth must belong to the leaf registry")
    choices = choices or PromptChoiceRegistry.from_registry(registry)
    result = check_stage1_choices(solution, choices=choices)
    if not result.format_valid:
        return Stage1Evaluation(None, False, False, False, result.errors)
    if not result.constraint_valid:
        return Stage1Evaluation(None, True, False, False, result.errors)
    assert result.decoded is not None
    return Stage1Evaluation(
        result.decoded, True, True, ground_truth in result.decoded, ()
    )


def evaluate_stage2(
    solution: str,
    *,
    ground_truth: str,
    candidates: Sequence[str],
    registry: LeafRegistry,
) -> Stage2Evaluation:
    """Evaluate membership and correctness, even when Stage 1 omitted the GT."""
    if ground_truth not in registry.ids:
        raise ValueError("ground_truth must belong to the leaf registry")
    if isinstance(candidates, (str, bytes)) or (
        len(candidates) != 5
        or len(set(candidates)) != 5
        or any(candidate not in registry.ids for candidate in candidates)
    ):
        raise ValueError("candidates must be 5 unique IDs from the leaf registry")
    result = check_stage2_output(solution, candidates=candidates)
    contract_valid = result.ok
    errors = () if contract_valid else result.errors
    return Stage2Evaluation(
        result.output.answer if result.format_valid else None,
        result.format_valid,
        contract_valid,
        contract_valid and result.output.answer == ground_truth,
        errors,
    )


def evaluate_stage2_choices(
    solution: str,
    *,
    ground_truth: str,
    candidates: Sequence[str],
    registry: LeafRegistry,
) -> Stage2Evaluation:
    """Evaluate a local-id Stage 2 output; decode BEFORE canonical logic.

    The model answers with a LOCAL bundle id ("1".."5" in candidate order);
    the returned prediction is the decoded canonical category_id. Consumes
    the shared choice-aware check (check_stage2_choices) so evaluation and
    reward share one choice validation implementation. Anything but an
    exact local id yields an explicit invalid result with no name or fuzzy
    fallback.
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
    if not result.format_valid:
        return Stage2Evaluation(None, False, False, False, result.errors)
    if not result.constraint_valid:
        assert isinstance(result.output, Stage2Output)
        return Stage2Evaluation(
            result.output.answer, True, False, False, result.errors
        )
    assert result.decoded is not None
    return Stage2Evaluation(
        result.decoded, True, True, result.decoded == ground_truth, ()
    )
