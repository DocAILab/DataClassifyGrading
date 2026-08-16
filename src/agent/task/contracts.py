"""Explicit task and leaf-category contracts used by SFT export/validation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class LeafCategory:
    category_id: str
    description: str = ""


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
            else:
                raise ValueError(f"invalid leaf registry category at index {index}")
            categories.append(LeafCategory(category_id, description))
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
