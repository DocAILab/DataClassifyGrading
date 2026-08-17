"""Algorithm-independent evaluation for two-stage leaf classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from agent.task import LeafRegistry
from agent.task.parser import (
    PredictionFormatError,
    parse_stage1_output,
    parse_stage2_output,
)


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
    """Evaluate format, candidate constraints, and Recall@5 separately."""
    if ground_truth not in registry.ids:
        raise ValueError("ground_truth must belong to the leaf registry")
    try:
        output = parse_stage1_output(solution)
    except PredictionFormatError as exc:
        return Stage1Evaluation(None, False, False, False, (str(exc),))

    errors: list[str] = []
    candidates = output.candidates
    if len(candidates) != 5:
        errors.append("stage1 prediction must contain exactly 5 candidates")
    if len(set(candidates)) != len(candidates):
        errors.append("stage1 candidates must be unique")
    if any(candidate not in registry.ids for candidate in candidates):
        errors.append("stage1 candidates must belong to the leaf registry")
    contract_valid = not errors
    return Stage1Evaluation(
        candidates,
        True,
        contract_valid,
        contract_valid and ground_truth in candidates,
        tuple(errors),
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
    try:
        output = parse_stage2_output(solution)
    except PredictionFormatError as exc:
        return Stage2Evaluation(None, False, False, False, (str(exc),))

    contract_valid = output.answer in candidates
    errors = () if contract_valid else ("stage2 answer must be one of the candidates",)
    return Stage2Evaluation(
        output.answer,
        True,
        contract_valid,
        contract_valid and output.answer == ground_truth,
        errors,
    )
