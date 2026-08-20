"""Canonical standard contracts (Phase 1).

Phase 0 frozen semantics respected here:
- ``sample_data_level`` (per-field label in processed/canonical) and
  ``standard_data_level`` (grade the classification/grading standard assigns
  to a category) are distinct and are NEVER merged or overwritten.
- ``standard_data_level`` is a category-level reference only; no natural-
  language semantics of L1..L4 are asserted here, and levels are not assumed
  equivalent across datasets.
- ``path`` stores the REAL source-hierarchy depth of the standard; empty
  levels are omitted (no invented padding).
- category_id continues the existing stable identity strategy so the
  standard stays joinable to the current registry/canonical targets.

Determinism: categories are canonicalized by sorted category_id and JSON is
written with sort_keys; ``fingerprint()`` is a sha256 over the canonical
categories payload (no timestamps, no machine-local paths).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

_WS_COLLAPSE_RE = re.compile(r"\s+")
_WS_REMOVE_RE = re.compile(r"\s+")
_TRAILING_CODE_RE = re.compile(
    r"\s*[\(\[（【]\s*([A-Za-z]+\d*(?:-\d+)*)\s*[\)\]）】]\s*$"
)

LEVELS = ("L1", "L2", "L3", "L4")

# Aliases accepted when normalizing a raw standard level value. Anything not
# covered here is kept raw and reported, never guessed.
_LEVEL_ALIASES: dict[str, str] = {
    "1": "L1", "2": "L2", "3": "L3", "4": "L4",
    "L1": "L1", "L2": "L2", "L3": "L3", "L4": "L4",
    "LEVEL1": "L1", "LEVEL2": "L2", "LEVEL3": "L3", "LEVEL4": "L4",
    "1级": "L1", "2级": "L2", "3级": "L3", "4级": "L4",
}


def clean(value: Any) -> str:
    """Collapse every whitespace run to a single space and strip."""
    if value is None:
        return ""
    return _WS_COLLAPSE_RE.sub(" ", str(value).strip())


def compact(value: Any) -> str:
    """Remove every whitespace character (identity seed; matches identity.py)."""
    if value is None:
        return ""
    return _WS_REMOVE_RE.sub("", str(value))


def strip_code(text: str) -> str:
    """Remove a trailing classification code such as （A1-1-1）, (A), 【A】."""
    return _TRAILING_CODE_RE.sub("", text).strip()


def normalize_standard_level(
    raw: Any,
) -> tuple[str | None, str]:
    """Return (canonical L1..L4 | None, cleaned raw value).

    Unparseable values (e.g. 'l', '3 4' from the finance guide) map to None
    and keep the raw text; callers must report them, never fix or guess.
    """
    text = clean(raw)
    if not text:
        return None, ""
    normalized = _LEVEL_ALIASES.get(text.upper())
    return (normalized, text) if normalized is not None else (None, text)


@dataclass(frozen=True)
class SourceRef:
    """Traceable origin of one standard category."""

    file: str = ""
    sheet: str = ""
    row: int | None = None

    def to_mapping(self) -> dict[str, Any]:
        return {"file": self.file, "sheet": self.sheet, "row": self.row}

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SourceRef":
        return cls(
            file=str(value.get("file", "") or ""),
            sheet=str(value.get("sheet", "") or ""),
            row=value.get("row"),
        )


@dataclass(frozen=True)
class StandardCategory:
    """One category of a canonical standard (the standard's own facts)."""

    category_id: str
    name: str
    path: tuple[str, ...] = ()
    description: str = ""
    code: str | None = None
    standard_data_level: str | None = None
    raw_level: str = ""
    content: str = ""
    source: SourceRef = field(default_factory=SourceRef)
    descriptions: tuple[str, ...] = ()

    def to_mapping(self) -> dict[str, Any]:
        mapping: dict[str, Any] = {
            "category_id": self.category_id,
            "name": self.name,
            "path": list(self.path),
            "description": self.description,
            "code": self.code,
            "standard_data_level": self.standard_data_level,
            "raw_level": self.raw_level,
            "source": self.source.to_mapping(),
        }
        if self.content:
            mapping["content"] = self.content
        if self.descriptions:
            mapping["descriptions"] = list(self.descriptions)
        return mapping

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "StandardCategory":
        source = value.get("source") or {}
        return cls(
            category_id=str(value.get("category_id", "") or ""),
            name=str(value.get("name", "") or ""),
            path=tuple(str(p) for p in value.get("path", ())),
            description=str(value.get("description", "") or ""),
            code=value.get("code"),
            standard_data_level=value.get("standard_data_level"),
            raw_level=str(value.get("raw_level", "") or ""),
            content=str(value.get("content", "") or ""),
            source=SourceRef.from_mapping(
                source if isinstance(source, Mapping) else {}
            ),
            descriptions=tuple(str(d) for d in value.get("descriptions", ())),
        )


@dataclass(frozen=True)
class CanonicalStandard:
    """The canonical standard for one dataset."""

    dataset: str
    id_strategy: str
    standard_source: SourceRef
    standard_name: str = ""
    categories: tuple[StandardCategory, ...] = ()

    def by_id(self) -> dict[str, StandardCategory]:
        return {category.category_id: category for category in self.categories}

    def to_mapping(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "id_strategy": self.id_strategy,
            "standard_name": self.standard_name,
            "standard_source": self.standard_source.to_mapping(),
            "fingerprint": self.fingerprint(),
            "categories": [category.to_mapping() for category in self.categories],
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CanonicalStandard":
        raw_categories = value.get("categories", ())
        categories = tuple(
            StandardCategory.from_mapping(item)
            for item in raw_categories
            if isinstance(item, Mapping)
        )
        source = value.get("standard_source") or {}
        return cls(
            dataset=str(value.get("dataset", "") or ""),
            id_strategy=str(value.get("id_strategy", "") or ""),
            standard_name=str(value.get("standard_name", "") or ""),
            standard_source=SourceRef.from_mapping(
                source if isinstance(source, Mapping) else {}
            ),
            categories=categories,
        )

    def fingerprint(self) -> str:
        payload = {
            "dataset": self.dataset,
            "id_strategy": self.id_strategy,
            "categories": [
                {k: v for k, v in category.to_mapping().items() if k != "source"}
                for category in sorted(self.categories, key=lambda c: c.category_id)
            ],
        }
        digest = hashlib.sha256()
        digest.update(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        )
        return digest.hexdigest()


@dataclass(frozen=True)
class StandardCategoryBuilder:
    """Deterministic canonical category from raw standard source fields.

    Kept as a small value object so build logic is trivially testable with
    plain dicts (no Excel, no IO).
    """

    category_id: str
    name: str
    path: tuple[str, ...]
    description: str = ""
    code: str | None = None
    raw_level: str = ""
    content: str = ""
    source_file: str = ""
    source_sheet: str = ""
    source_row: int | None = None

    def build(self) -> StandardCategory:
        level, _ = normalize_standard_level(self.raw_level)  # level kept, raw kept
        return StandardCategory(
            category_id=self.category_id,
            name=self.name,
            path=self.path,
            description=self.description,
            code=self.code,
            standard_data_level=level,
            raw_level=clean(self.raw_level),
            content=self.content,
            source=SourceRef(
                file=self.source_file, sheet=self.source_sheet, row=self.source_row
            ),
        )


__all__ = [
    "LEVELS",
    "clean",
    "compact",
    "strip_code",
    "normalize_standard_level",
    "SourceRef",
    "StandardCategory",
    "CanonicalStandard",
    "StandardCategoryBuilder",
]
