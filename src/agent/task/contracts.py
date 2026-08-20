"""Explicit task and leaf-category contracts used by SFT export/validation.

Stage 2 additions (canonical unified data contract):
- LeafCategory now carries name / path / code in addition to category_id and
  description. category_id is the only identity that training depends on;
  name is NOT assumed unique (same leaf name can exist under different
  parent paths, e.g. finance '基本信息' under 3 parents).
- CorpusCategory is the canonical corpus category contract; it allows
  missing descriptions and multiple examples per category.
- SampleTarget is the canonical per-sample training target; the raw
  classification levels are kept untouched as provenance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
    - path: ancestor names from root to this leaf, e.g. shougang
      ("生产数据域", "生产合同（订单）", "合同归并") or finance
      ("业务", "交易信息", "交易通用信息", "交易清结算信息"). Empty levels are
      omitted; the tuple is provenance, never an ID.
    - code: optional stable code from a standard (e.g. guanji "A1-1-1");
      treated as an opaque identity, digit groups are NOT interpreted as
      level_2 / level_3.
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
class CorpusScopedAnnotation:
    """Task-layer view of a standard scoped annotation attached to a category.

    Mirrors agent.standards.contracts.ScopedAnnotation without importing the
    standards package from the task layer (avoids package-init cycles).
    """

    annotation_id: str
    type: str
    text: str
    source_cell: str = ""
    merged_range: str | None = None
    start_row: int | None = None
    end_row: int | None = None
    applies_to_standard_entry_ids: tuple[str, ...] = ()

    def to_mapping(self) -> dict[str, Any]:
        return {
            "annotation_id": self.annotation_id,
            "type": self.type,
            "text": self.text,
            "source_cell": self.source_cell,
            "merged_range": self.merged_range,
            "start_row": self.start_row,
            "end_row": self.end_row,
            "applies_to_standard_entry_ids": list(self.applies_to_standard_entry_ids),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CorpusScopedAnnotation":
        return cls(
            annotation_id=str(value.get("annotation_id", "") or ""),
            type=str(value.get("type", "") or ""),
            text=str(value.get("text", "") or ""),
            source_cell=str(value.get("source_cell", "") or ""),
            merged_range=value.get("merged_range"),
            start_row=value.get("start_row"),
            end_row=value.get("end_row"),
            applies_to_standard_entry_ids=tuple(
                str(item) for item in value.get("applies_to_standard_entry_ids", ())
            ),
        )


@dataclass(frozen=True)
class StandardEntryView:
    """One standard entry projected onto a corpus category (full facts).

    The Phase-1 canonical standard is LOSSESS: a training category_id may be
    backed by several standard entries (e.g. finance 5× 基本信息 under five
    三级). This view keeps EVERY entry (id, real path, its own description,
    grading reference, hierarchy definitions) — never a first-only collapse.
    None of this is exposed to the Stage 2 prompt this phase.
    """

    standard_entry_id: str
    name: str = ""
    path: tuple[str, ...] = ()
    description: str = ""
    raw_level: str = ""
    standard_data_level: str | None = None
    content: str = ""
    code: str | None = None
    source: Mapping[str, Any] = field(default_factory=dict)
    raw_fields: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    def to_mapping(self) -> dict[str, Any]:
        mapping: dict[str, Any] = {
            "standard_entry_id": self.standard_entry_id,
            "name": self.name,
            "path": list(self.path),
            "description": self.description,
            "raw_level": self.raw_level,
            "standard_data_level": self.standard_data_level,
            "source": dict(self.source),
        }
        if self.content:
            mapping["content"] = self.content
        if self.code is not None:
            mapping["code"] = self.code
        if self.raw_fields:
            mapping["raw_fields"] = {
                key: dict(item) for key, item in sorted(self.raw_fields.items())
            }
        return mapping

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "StandardEntryView":
        raw_fields = value.get("raw_fields") or {}
        source = value.get("source") or {}
        return cls(
            standard_entry_id=str(value.get("standard_entry_id", "") or ""),
            name=str(value.get("name", "") or ""),
            path=tuple(str(p) for p in value.get("path", ())),
            description=str(value.get("description", "") or ""),
            raw_level=str(value.get("raw_level", "") or ""),
            standard_data_level=value.get("standard_data_level"),
            content=str(value.get("content", "") or ""),
            code=value.get("code"),
            source=(
                {str(k): v for k, v in source.items()}
                if isinstance(source, Mapping)
                else {}
            ),
            raw_fields=(
                {str(k): dict(v) for k, v in raw_fields.items()}
                if isinstance(raw_fields, Mapping)
                else {}
            ),
        )


@dataclass(frozen=True)
class CorpusCategory:
    """Canonical corpus category (one category may own several documents).

    Allows:
    - one primary description plus additional descriptions of the same
      category (descriptions tuple), e.g. several guide documents that
      describe the same leaf;
    - genuine examples (examples tuple) kept semantically separate from
      descriptions;
    - missing description (pers_info corpus is incomplete),
    - an incomplete corpus overall (no global completeness invariant).

    Phase-2 (from CanonicalStandard): the category may be backed by multiple
    standard entries (``standard_entry_ids`` / ``standard_entries``) and carry
    the scoped grading annotations that apply to it. These fields are the
    Stage-2-knowledge fact layer and are NOT emitted into the Stage 2 prompt
    this phase.
    """

    category_id: str
    name: str
    description: str = ""
    descriptions: tuple[str, ...] = ()
    path: tuple[str, ...] = ()
    code: str | None = None
    examples: tuple[str, ...] = ()
    standard_entry_ids: tuple[str, ...] = ()
    standard_entries: tuple[StandardEntryView, ...] = ()
    scoped_annotations: tuple[CorpusScopedAnnotation, ...] = ()

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
