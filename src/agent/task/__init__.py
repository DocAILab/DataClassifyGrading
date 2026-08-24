"""Algorithm-independent contracts and prompts for leaf classification."""

from .contracts import (
    CorpusCategory,
    LeafCategory,
    LeafRegistry,
    SampleTarget,
    TaskConfig,
)
from .canonical_builder import (
    SCHEMA_VERSION as CANONICAL_SCHEMA_VERSION,
    CanonicalBuildResult,
    prepare_canonical_dataset,
    resolve_record as resolve_canonical_record,
)
from .assets import ClassificationAssets, load_corpus_categories
from .dataset_config import (
    DEFAULT_PATH_FIELDS,
    ID_STRATEGIES,
    REGISTRY_DERIVATIONS,
    DatasetConfig,
    load_dataset_configs,
)
from .identity import code_leaf_map, compact, leaf_registry_from_corpus, qualified_category_id
from .parser import (
    ChoiceParseResult,
    ParseResult,
    PredictionFormatError,
    Stage1Output,
    Stage2Output,
    check_stage1_choices,
    check_stage1_output,
    check_stage2_choices,
    check_stage2_output,
    parse_stage1_output,
    parse_stage2_output,
)
from .prompt_choices import (
    PromptChoice,
    PromptChoiceError,
    PromptChoiceRegistry,
    decode_stage2_answer,
    encode_stage2_answer,
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
    "ClassificationAssets",
    "load_corpus_categories",
    "DatasetConfig",
    "load_dataset_configs",
    "ID_STRATEGIES",
    "DEFAULT_PATH_FIELDS",
    "REGISTRY_DERIVATIONS",
    "CANONICAL_SCHEMA_VERSION",
    "CanonicalBuildResult",
    "prepare_canonical_dataset",
    "resolve_canonical_record",
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
    "ChoiceParseResult",
    "check_stage1_output",
    "check_stage2_output",
    "check_stage1_choices",
    "check_stage2_choices",
    "parse_stage1_output",
    "parse_stage2_output",
    "PromptChoice",
    "PromptChoiceError",
    "PromptChoiceRegistry",
    "decode_stage2_answer",
    "encode_stage2_answer",
    "Prompt",
    "build_stage1_prompt",
    "build_stage2_prompt",
    "stage1_answer",
    "stage2_answer",
]
