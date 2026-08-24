"""Runtime-loaded configuration for canonical target resolution.

Dataset-specific strategies are not embedded in source code. Callers load them
from an explicit local JSON file or construct ``DatasetConfig`` directly.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

ID_STRATEGIES = ("code", "path")
DEFAULT_PATH_FIELDS = ("level_1", "level_2", "level_3", "level_4")
# How the leaf universe behind this dataset's registry was derived. Pure
# documentation/audit metadata: it never changes resolution semantics.
REGISTRY_DERIVATIONS = (
    "standard",          # derived from a classification standard / corpus
    "shared-standard",   # reuses another dataset's standard via registry_source
    "dataset-universe",  # no standard exists; registry taken from the full data leaf space
)


@dataclass(frozen=True)
class DatasetConfig:
    """Configuration for resolving one dataset's canonical target identity."""

    dataset: str
    leaf_level: str = "level_4"
    id_strategy: str = "path"
    path_fields: tuple[str, ...] = DEFAULT_PATH_FIELDS
    identity_fields: tuple[str, ...] = ()
    placeholder_labels: tuple[str, ...] = ()
    registry_source: str | None = None
    # Category ids deliberately kept out of the active registry/corpus
    # (e.g. standard entries absent from the legacy Stage-1 universe).
    # The canonical builder treats a resolved hit on any of these as a
    # configuration error instead of silently training on them.
    projection_excluded_category_ids: tuple[str, ...] = ()
    registry_derivation: str = "standard"

    def __post_init__(self) -> None:
        if not self.dataset.strip():
            raise ValueError("dataset name must be non-empty")
        if self.id_strategy not in ID_STRATEGIES:
            raise ValueError(
                f"id_strategy must be one of {ID_STRATEGIES}, got {self.id_strategy!r}"
            )
        if not self.path_fields or any(not field for field in self.path_fields):
            raise ValueError("path_fields must contain non-empty strings")
        if len(set(self.path_fields)) != len(self.path_fields):
            raise ValueError("path_fields must be unique")
        if self.leaf_level not in self.path_fields:
            raise ValueError(
                f"leaf_level {self.leaf_level!r} must be one of path_fields "
                f"{self.path_fields}"
            )
        if self.identity_fields:
            if len(set(self.identity_fields)) != len(self.identity_fields):
                raise ValueError("identity_fields must be unique")
            unknown = set(self.identity_fields) - set(self.path_fields)
            if unknown:
                raise ValueError(
                    f"identity_fields must be a subset of path_fields, got {unknown}"
                )
            if self.leaf_level not in self.identity_fields:
                raise ValueError("leaf_level must participate in identity_fields")
        if self.registry_source == self.dataset:
            raise ValueError("registry_source must not point at the dataset itself")
        if len(set(self.projection_excluded_category_ids)) != len(
            self.projection_excluded_category_ids
        ):
            raise ValueError("projection_excluded_category_ids must be unique")
        if self.registry_derivation not in REGISTRY_DERIVATIONS:
            raise ValueError(
                f"registry_derivation must be one of {REGISTRY_DERIVATIONS}, "
                f"got {self.registry_derivation!r}"
            )
        if (
            self.registry_source is None
            and self.registry_derivation == "shared-standard"
        ):
            raise ValueError(
                'registry_derivation "shared-standard" requires registry_source'
            )

    def to_mapping(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "dataset": self.dataset,
            "leaf_level": self.leaf_level,
            "id_strategy": self.id_strategy,
            "path_fields": list(self.path_fields),
            "placeholder_labels": list(self.placeholder_labels),
            "projection_excluded_category_ids": list(
                self.projection_excluded_category_ids
            ),
            "registry_derivation": self.registry_derivation,
        }
        if self.identity_fields:
            result["identity_fields"] = list(self.identity_fields)
        if self.registry_source is not None:
            result["registry_source"] = self.registry_source
        return result

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "DatasetConfig":
        dataset = _required_string(value.get("dataset"), "dataset")
        path_fields = _string_tuple(
            value.get("path_fields", DEFAULT_PATH_FIELDS), "path_fields"
        )
        identity_fields = _string_tuple(
            value.get("identity_fields", ()), "identity_fields"
        )
        placeholders = _string_tuple(
            value.get("placeholder_labels", ()), "placeholder_labels"
        )
        excluded = _string_tuple(
            value.get("projection_excluded_category_ids", ()),
            "projection_excluded_category_ids",
        )
        derivation = value.get("registry_derivation", "standard")
        if not isinstance(derivation, str):
            raise ValueError("registry_derivation must be a string")
        source = value.get("registry_source")
        if source is not None and not isinstance(source, str):
            raise ValueError("registry_source must be a string or null")
        return cls(
            dataset=dataset,
            leaf_level=_required_string(value.get("leaf_level", "level_4"), "leaf_level"),
            id_strategy=_required_string(value.get("id_strategy", "path"), "id_strategy"),
            path_fields=path_fields,
            identity_fields=identity_fields,
            placeholder_labels=placeholders,
            registry_source=None if source is None else source.strip(),
            projection_excluded_category_ids=excluded,
            registry_derivation=derivation.strip(),
        )

    @classmethod
    def from_path(cls, path: str | Path) -> "DatasetConfig":
        source = Path(path)
        with source.open(encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, Mapping):
            raise ValueError("dataset config must be a JSON object")
        return cls.from_mapping(value)


def load_dataset_configs(path: str | Path) -> dict[str, DatasetConfig]:
    """Load dataset configurations from a local JSON array or datasets object."""
    source = Path(path)
    with source.open(encoding="utf-8") as handle:
        value = json.load(handle)
    raw: Any = value.get("datasets") if isinstance(value, Mapping) else value
    if isinstance(raw, Mapping):
        items = []
        for name, config in raw.items():
            if not isinstance(config, Mapping):
                raise ValueError(f"dataset config {name!r} must be an object")
            item = dict(config)
            item.setdefault("dataset", name)
            items.append(item)
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        items = list(raw)
    else:
        raise ValueError("dataset config file must contain a datasets object or array")

    configs: dict[str, DatasetConfig] = {}
    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            raise ValueError(f"invalid dataset config at index {index}")
        config = DatasetConfig.from_mapping(item)
        if config.dataset in configs:
            raise ValueError(f"duplicate dataset config: {config.dataset}")
        configs[config.dataset] = config
    if not configs:
        raise ValueError("dataset config file must contain at least one dataset")
    return configs


def _required_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _string_tuple(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field} must be an array of strings")
    if not all(isinstance(item, str) and item.strip() for item in value):
        raise ValueError(f"{field} must contain non-empty strings")
    return tuple(item.strip() for item in value)


__all__ = [
    "DatasetConfig",
    "load_dataset_configs",
    "ID_STRATEGIES",
    "DEFAULT_PATH_FIELDS",
    "REGISTRY_DERIVATIONS",
]
