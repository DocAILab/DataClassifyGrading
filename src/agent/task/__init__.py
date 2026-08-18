"""Algorithm-independent contracts and prompts for leaf classification."""

from .contracts import (
    CorpusCategory,
    LeafCategory,
    LeafRegistry,
    SampleTarget,
    TaskConfig,
)
from .dataset_config import BUILTIN_DATASET_CONFIGS, DatasetConfig
from .identity import build_leaf_registry, code_leaf_map, compact, stable_category_id
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
from .resolver import (
    ClassificationTargetResolver,
    TargetResolver,
    resolve_all,
)

__all__ = [
    "LeafCategory",
    "LeafRegistry",
    "CorpusCategory",
    "SampleTarget",
    "TaskConfig",
    "DatasetConfig",
    "BUILTIN_DATASET_CONFIGS",
    "TargetResolver",
    "ClassificationTargetResolver",
    "resolve_all",
    "stable_category_id",
    "code_leaf_map",
    "build_leaf_registry",
    "compact",
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
