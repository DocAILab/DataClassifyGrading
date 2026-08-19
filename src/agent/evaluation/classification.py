"""Algorithm-independent evaluation for two-stage leaf classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from agent.task import LeafRegistry
from agent.task.parser import check_stage1_output, check_stage2_output


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
