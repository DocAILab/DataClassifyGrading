"""Task and leaf-category contracts used by classification and training.

``category_id`` is the only canonical identity. Display names may repeat
under different parent paths, corpus descriptions may be absent, and raw
classification fields remain provenance rather than fallback labels.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class LeafCategory:
    """One leaf category of a dataset LeafRegistry.

    - category_id: unique within one LeafRegistry; the only identity the
      training ground truth depends on.
    - name: human leaf name; NOT assumed unique across categories.
    - description: optional free text (may be empty when no corpus exists).
    - path: ancestor names from root to this leaf. Empty levels are omitted;
      the tuple is provenance, never an ID.
    - code: optional stable code supplied by a local classification standard;
      treated as an opaque identity whose internal segments are not parsed.
    """

    category_id: str
    description: str = ""
    name: str = ""
    path: tuple[str, ...] = ()
    code: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            # Backward compatibility: registries without an explicit name use
            # the category_id as display name.
            object.__setattr__(self, "name", self.category_id)


@dataclass(frozen=True)
class CorpusCategory:
    """Canonical corpus category (one category may own several documents).

    Allows:
    - one primary description plus additional descriptions of the same
      category (descriptions tuple);
    - genuine examples (examples tuple) kept semantically separate from
      descriptions;
    - a missing description,
    - an incomplete corpus overall (no global completeness invariant).
    """

    category_id: str
    name: str
    description: str = ""
    descriptions: tuple[str, ...] = ()
    path: tuple[str, ...] = ()
    code: str | None = None
    examples: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.category_id.strip():
            raise ValueError("corpus category_id must be non-empty")
        if not self.name.strip():
            raise ValueError("corpus category name must be non-empty")


@dataclass(frozen=True)
class SampleTarget:
    """Canonical per-sample training target derived from a processed record.

    The record's classification.level_1..level_4 are left untouched as
    provenance; the training ground truth is target.category_id only.
    """

    leaf_level: str
    leaf_name: str
    category_id: str
    category_path: tuple[str, ...]


@dataclass(frozen=True)
class LeafRegistry:
    categories: tuple[LeafCategory, ...]

    def __post_init__(self) -> None:
        ids = [category.category_id for category in self.categories]
        if len(self.categories) < 5:
            raise ValueError("leaf registry must contain at least 5 categories")
        if any(not item for item in ids):
            raise ValueError("leaf registry category_id must be non-empty")
        if len(set(ids)) != len(ids):
            raise ValueError("leaf registry category_id values must be unique")

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(category.category_id for category in self.categories)

    def get(self, category_id: str) -> LeafCategory:
        for category in self.categories:
            if category.category_id == category_id:
                return category
        raise KeyError(category_id)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | Sequence[Any]) -> "LeafRegistry":
        raw = value.get("categories") if isinstance(value, Mapping) else value
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            raise ValueError("leaf registry must contain a categories array")
        categories: list[LeafCategory] = []
        for index, item in enumerate(raw):
            if isinstance(item, str):
                category_id, description = item.strip(), ""
            elif isinstance(item, Mapping):
                category_id = str(item.get("category_id", "")).strip()
                raw_description = item.get("description", "")
                description = "" if raw_description is None else str(raw_description).strip()
                raw_name = item.get("name", "")
                name = "" if raw_name is None else str(raw_name).strip()
                raw_path = item.get("path", ())
                path = tuple(
                    str(part).strip() for part in raw_path
                ) if raw_path else ()
                raw_code = item.get("code")
                code = None if raw_code is None else str(raw_code).strip()
                categories.append(
                    LeafCategory(
                        category_id=category_id,
                        description=description,
                        name=name,
                        path=path,
                        code=code,
                    )
                )
                continue
            else:
                raise ValueError(f"invalid leaf registry category at index {index}")
            categories.append(LeafCategory(category_id=category_id, description=description))
        return cls(tuple(categories))

    @classmethod
    def from_path(cls, path: str | Path) -> "LeafRegistry":
        source = Path(path)
        with source.open(encoding="utf-8") as handle:
            return cls.from_mapping(json.load(handle))


@dataclass(frozen=True)
class TaskConfig:
    """Prompt visibility is intentionally explicit; there are no implicit metadata fields."""

    metadata_fields: tuple[str, ...]
    task_name: str = "data_classification"

    def __post_init__(self) -> None:
        if not self.metadata_fields:
            raise ValueError("task config metadata_fields must not be empty")
        if any(not isinstance(field, str) or not field.strip() for field in self.metadata_fields):
            raise ValueError("task config metadata_fields must contain non-empty strings")
        if len(set(self.metadata_fields)) != len(self.metadata_fields):
            raise ValueError("task config metadata_fields must be unique")
        if not isinstance(self.task_name, str) or not self.task_name.strip():
            raise ValueError("task config task_name must be a non-empty string")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TaskConfig":
        fields = value.get("metadata_fields")
        if not isinstance(fields, Sequence) or isinstance(fields, (str, bytes)):
            raise ValueError("task config requires metadata_fields array")
        if not all(isinstance(field, str) for field in fields):
            raise ValueError("task config metadata_fields must be strings")
        raw_task_name = value.get("task_name", cls.task_name)
        if not isinstance(raw_task_name, str):
            raise ValueError("task config task_name must be a string")
        return cls(
            tuple(field.strip() for field in fields),
            raw_task_name.strip(),
        )

    @classmethod
    def from_path(cls, path: str | Path) -> "TaskConfig":
        source = Path(path)
        with source.open(encoding="utf-8") as handle:
            return cls.from_mapping(json.load(handle))
