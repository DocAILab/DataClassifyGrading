"""Per-dataset configuration for the canonical training target.

Decision basis: artifacts/data_alignment_report (stage 1). Each dataset gets
an explicit category_id strategy instead of a single global rule:

- shougang: guanji codes are stable and unique (233 codes, letter prefix
  proven to map 1:1 to level_1; digit groups are NOT interpreted as
  level_2/level_3 — the code is an opaque identity).
- infra: a strict subset of shougang (4/4 leaves, same codes); it reuses the
  shougang registry via registry_source instead of building a 4-category one.
- finance: the standard is a 3-segment path while the dataset has 4 levels;
  no reliable code exists, so the ID is a deterministic hash of the full
  4-level path (empty levels kept as separators), which guarantees that the
  same leaf name under different parents gets different category_ids.
- pers_info: only level_4 exists (18 unique labels), corpus covers 4/18, so
  the schema must allow targets with no corpus description; IDs come from the
  deterministic path hash of the single populated level.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

ID_STRATEGIES = ("code", "path_hash")

DEFAULT_PATH_FIELDS = ("level_1", "level_2", "level_3", "level_4")


@dataclass(frozen=True)
class DatasetConfig:
    """Explicit, per-dataset configuration for target generation.

    Attributes:
      dataset: dataset name, e.g. "finance".
      leaf_level: the canonical leaf classification level, e.g. "level_4".
      id_strategy: "code" (stable corpus code, e.g. guanji A1-1-1) or
        "path_hash" (deterministic hash of the full path fields).
      path_fields: classification fields used to build the category path and
        the hash seed, in order from root to leaf.
      placeholder_labels: labels that are not real categories (e.g. shougang
        "——" placeholder); records with such a leaf resolve to no target.
      registry_source: optional name of another dataset whose LeafRegistry
        this dataset reuses (infra -> shougang); None builds its own.
    """

    dataset: str
    leaf_level: str = "level_4"
    id_strategy: str = "path_hash"
    path_fields: tuple[str, ...] = DEFAULT_PATH_FIELDS
    placeholder_labels: tuple[str, ...] = ()
    registry_source: str | None = None

    def __post_init__(self) -> None:
        if not self.dataset.strip():
            raise ValueError("dataset name must be non-empty")
        if self.id_strategy not in ID_STRATEGIES:
            raise ValueError(
                f"id_strategy must be one of {ID_STRATEGIES}, got {self.id_strategy!r}"
            )
        if not self.path_fields:
            raise ValueError("path_fields must not be empty")
        if len(set(self.path_fields)) != len(self.path_fields):
            raise ValueError("path_fields must be unique")
        if self.leaf_level not in self.path_fields:
            raise ValueError(
                f"leaf_level {self.leaf_level!r} must be one of path_fields "
                f"{self.path_fields}"
            )
        if self.registry_source == self.dataset:
            raise ValueError("registry_source must not point at the dataset itself")

    def to_mapping(self) -> dict[str, Any]:
        mapping: dict[str, Any] = {
            "dataset": self.dataset,
            "leaf_level": self.leaf_level,
            "id_strategy": self.id_strategy,
            "path_fields": list(self.path_fields),
            "placeholder_labels": list(self.placeholder_labels),
        }
        if self.registry_source is not None:
            mapping["registry_source"] = self.registry_source
        return mapping

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "DatasetConfig":
        raw_dataset = value.get("dataset")
        if not isinstance(raw_dataset, str) or not raw_dataset.strip():
            raise ValueError("dataset config requires a non-empty dataset name")
        raw_fields = value.get("path_fields", DEFAULT_PATH_FIELDS)
        if not isinstance(raw_fields, (list, tuple)) or not all(
            isinstance(field, str) for field in raw_fields
        ):
            raise ValueError("dataset config path_fields must be a list of strings")
        raw_placeholders = value.get("placeholder_labels", ())
        if not isinstance(raw_placeholders, (list, tuple)) or not all(
            isinstance(label, str) for label in raw_placeholders
        ):
            raise ValueError("dataset config placeholder_labels must be a list of strings")
        raw_source = value.get("registry_source")
        if raw_source is not None and not isinstance(raw_source, str):
            raise ValueError("dataset config registry_source must be a string or null")
        return cls(
            dataset=raw_dataset.strip(),
            leaf_level=str(value.get("leaf_level", "level_4")).strip(),
            id_strategy=str(value.get("id_strategy", "path_hash")).strip(),
            path_fields=tuple(field.strip() for field in raw_fields),
            placeholder_labels=tuple(label.strip() for label in raw_placeholders),
            registry_source=None if raw_source is None else raw_source.strip(),
        )

    @classmethod
    def from_path(cls, path: str | Path) -> "DatasetConfig":
        source = Path(path)
        with source.open(encoding="utf-8") as handle:
            return cls.from_mapping(json.load(handle))


# Built-in per-dataset strategies derived from the stage-1 alignment report.
BUILTIN_DATASET_CONFIGS: dict[str, DatasetConfig] = {
    "shougang": DatasetConfig(
        dataset="shougang",
        id_strategy="code",
        placeholder_labels=("——",),
    ),
    "infra": DatasetConfig(
        dataset="infra",
        id_strategy="code",
        registry_source="shougang",
    ),
    "finance": DatasetConfig(dataset="finance"),
    "pers_info": DatasetConfig(dataset="pers_info"),
}

__all__ = [
    "DatasetConfig",
    "BUILTIN_DATASET_CONFIGS",
    "ID_STRATEGIES",
    "DEFAULT_PATH_FIELDS",
]
