"""Single source of truth for the formal training-release scope."""

from __future__ import annotations

FORMAL_DATASETS = ("shougang",)
FORMAL_DATASET_SET = frozenset(FORMAL_DATASETS)
FORMAL_RELEASE_NAME = "shougang"
FORMAL_RELEASE_FORMAT = "dataclassify-shougang-release-v1"
FORMAL_SAMPLING_POLICY = "single-dataset passthrough"

__all__ = [
    "FORMAL_DATASETS",
    "FORMAL_DATASET_SET",
    "FORMAL_RELEASE_FORMAT",
    "FORMAL_RELEASE_NAME",
    "FORMAL_SAMPLING_POLICY",
]
