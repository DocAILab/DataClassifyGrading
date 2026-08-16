"""Reusable evaluation interfaces for SFT tests and future RL rewards."""

from .classification import (
    Stage1Evaluation,
    Stage2Evaluation,
    evaluate_stage1,
    evaluate_stage2,
)

__all__ = [
    "Stage1Evaluation",
    "Stage2Evaluation",
    "evaluate_stage1",
    "evaluate_stage2",
]
