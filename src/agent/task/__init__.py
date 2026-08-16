"""Algorithm-independent contracts and prompts for leaf classification."""

from .contracts import LeafCategory, LeafRegistry, TaskConfig
from .parser import (
    PredictionFormatError,
    Stage1Output,
    Stage2Output,
    parse_stage1_output,
    parse_stage2_output,
)
from .prompts import (
    Prompt,
    build_stage1_prompt,
    build_stage2_prompt,
    stage1_answer,
    stage2_answer,
)

__all__ = [
    "LeafCategory",
    "LeafRegistry",
    "TaskConfig",
    "PredictionFormatError",
    "Stage1Output",
    "Stage2Output",
    "parse_stage1_output",
    "parse_stage2_output",
    "Prompt",
    "build_stage1_prompt",
    "build_stage2_prompt",
    "stage1_answer",
    "stage2_answer",
]
