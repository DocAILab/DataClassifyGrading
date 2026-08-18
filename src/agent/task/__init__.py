"""Algorithm-independent contracts and prompts for leaf classification."""

from .contracts import (
    CorpusCategory,
    LeafCategory,
    LeafRegistry,
    SampleTarget,
    TaskConfig,
)
from .dataset_config import BUILTIN_DATASET_CONFIGS, DatasetConfig
from .identity import code_leaf_map, compact, leaf_registry_from_corpus, qualified_category_id
from .parser import (
    ParseResult,
    PredictionFormatError,
    Stage1Output,
    Stage2Output,
    check_stage1_output,
    check_stage2_output,
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
    ResolutionResult,
    ResolutionStatus,
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
    "ResolutionStatus",
    "ResolutionResult",
    "resolve_all",
    "qualified_category_id",
    "code_leaf_map",
    "leaf_registry_from_corpus",
    "compact",
    "PredictionFormatError",
    "Stage1Output",
    "Stage2Output",
    "ParseResult",
    "check_stage1_output",
    "check_stage2_output",
    "parse_stage1_output",
    "parse_stage2_output",
    "Prompt",
    "build_stage1_prompt",
    "build_stage2_prompt",
    "stage1_answer",
    "stage2_answer",
]
