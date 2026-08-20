"""Canonical standard layer (Phase 1).

Lossless, auditable, reproducible canonical form of the original
classification/grading standards. Distinct from sample labels: see
docs/design/data_level_design.md (Phase 0).
"""

from .contracts import (
    LEVELS,
    CanonicalStandard,
    SourceRef,
    StandardCategory,
    StandardCategoryBuilder,
    clean,
    compact,
    normalize_standard_level,
    strip_code,
)
from .sources import (
    RawEntry,
    ReaderResult,
    read_finance_standard_guide,
    read_guanji_catalog,
)
from .build import (
    BuildIssue,
    StandardBuildReport,
    build_finance_standard,
    build_shougang_standard,
    resolve_standard_dataset,
)
from .align import (
    align_dataset_to_standard,
    load_canonical_records,
)

__all__ = [
    "LEVELS",
    "CanonicalStandard",
    "SourceRef",
    "StandardCategory",
    "StandardCategoryBuilder",
    "clean",
    "compact",
    "normalize_standard_level",
    "strip_code",
    "RawEntry",
    "ReaderResult",
    "read_finance_standard_guide",
    "read_guanji_catalog",
    "BuildIssue",
    "StandardBuildReport",
    "build_finance_standard",
    "build_shougang_standard",
    "resolve_standard_dataset",
    "align_dataset_to_standard",
    "load_canonical_records",
]
