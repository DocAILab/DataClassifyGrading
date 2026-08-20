"""Canonical standard builders (Phase 1).

Turn raw standard rows (from ``sources``) into a lossless, auditable
``CanonicalStandard``:
- category_id continues the existing stable identity strategy: finance uses
  the path-qualified L1/L2-leaf identity (level_3 excluded, matching
  DatasetConfig.identity_fields), shougang uses the guanji code.
- path stores the TRUE source-hierarchy depth (empty 三级子类 omitted — no
  invented padding).
- grading columns are normalized to L1..L4 with ``normalize_standard_level``;
  unparseable values are reported and kept raw, never fixed or guessed.
- placeholder / malformed rows (shougang "——", NaN, missing code) are
  skipped and reported in the build report — never silently repaired.

Builds are deterministic for identical input (sorted categories / issues);
the CLI writes every artifact only after all datasets build and align
successfully (fail-fast) and refuses to overwrite without --overwrite.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from agent.task.identity import qualified_category_id
from agent.standards.contracts import (
    CanonicalStandard,
    SourceRef,
    StandardCategory,
    StandardCategoryBuilder,
    clean,
    compact,
    normalize_standard_level,
    strip_code,
)

_PLACEHOLDER_NAMES = {"——", "nan", "none", "-", ""}
_TRAILING_CODE_RE = re.compile(
    r"[\(\[（【]\s*([A-Za-z]+\d*(?:-\d+)*)\s*[\)\]）】]\s*$"
)


@dataclass(frozen=True)
class BuildIssue:
    kind: str
    detail: str

    def to_mapping(self) -> dict[str, str]:
        return {"kind": self.kind, "detail": self.detail}


@dataclass
class StandardBuildReport:
    dataset: str
    id_strategy: str
    standard_name: str
    source_file: str
    source_sheet: str
    entries_read: int = 0
    categories_out: int = 0
    aggregated: dict[str, int] = field(default_factory=dict)
    level_distribution: dict[str, int] = field(default_factory=dict)
    issues: list[BuildIssue] = field(default_factory=list)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "id_strategy": self.id_strategy,
            "standard_name": self.standard_name,
            "source_file": self.source_file,
            "source_sheet": self.source_sheet,
            "entries_read": self.entries_read,
            "categories_out": self.categories_out,
            "aggregated": dict(self.aggregated),
            "level_distribution": dict(self.level_distribution),
            "issues": sorted(
                (issue.to_mapping() for issue in self.issues),
                key=lambda i: (i["kind"], i["detail"]),
            ),
        }


def _finalize_report_levels(report: StandardBuildReport, categories: Sequence[StandardCategory]) -> None:
    """Level distribution over the FINAL aggregated categories (not raw rows).

    Unparseable / missing levels are counted under an empty key.
    """
    counter: Counter[str] = Counter()
    for category in categories:
        counter[category.standard_data_level or "''"] += 1
    report.level_distribution = dict(sorted(counter.items()))


def _add_issue(report: StandardBuildReport, kind: str, detail: str) -> None:
    report.issues.append(BuildIssue(kind=kind, detail=detail))


def _dedupe_path(parts: Sequence[str]) -> tuple[str, ...]:
    """Drop empty parts and consecutive duplicates (a leaf living at level 3
    would otherwise repeat its name in the path)."""
    result: list[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if result and result[-1] == part:
            continue
        result.append(part)
    return tuple(result)


def _aggregate(
    categories: Iterable[StandardCategory],
    report: StandardBuildReport,
) -> list[StandardCategory]:
    """Merge rows that share a category_id (lossless: extras go to
    ``descriptions``). Reports level conflicts within one id, never fixes."""
    counts: Counter[str] = Counter()
    by_id: dict[str, StandardCategory] = {}
    collisions: list[tuple[str, str, str, str]] = []  # id, extra_level, row, primary_level
    for category in categories:
        counts[category.category_id] += 1
        existing = by_id.get(category.category_id)
        if existing is None:
            by_id[category.category_id] = category
            continue
        if (
            category.standard_data_level
            and existing.standard_data_level
            and category.standard_data_level != existing.standard_data_level
        ):
            collisions.append(
                (
                    category.category_id,
                    category.standard_data_level,
                    str(category.source.row),
                    existing.standard_data_level,
                )
            )
        by_id[category.category_id] = StandardCategory(
            category_id=existing.category_id,
            name=existing.name,
            path=existing.path,
            description=existing.description,
            code=existing.code,
            standard_data_level=existing.standard_data_level,
            raw_level=existing.raw_level,
            content=existing.content,
            source=existing.source,
            descriptions=existing.descriptions
            + ((category.description,) if category.description else ()),
        )
    repeated = {id_: n for id_, n in counts.items() if n > 1}
    report.aggregated = {
        "kinds": len(repeated),
        "instances": sum(n - 1 for n in repeated.values()),
    }
    for category_id, extra, row, primary in sorted(collisions):
        _add_issue(
            report,
            "level_conflict_within_category",
            f"category {category_id!r}: row {row} level {extra!r} differs from "
            f"primary {primary!r} (kept primary, not fixed)",
        )
    # deterministic, even when the input order varies
    return sorted(by_id.values(), key=lambda c: c.category_id)


def build_finance_standard(
    entries: Sequence[Any],
    *,
    source_file: str,
    source_sheet: str,
    dataset: str = "finance",
    standard_name: str = "金融行业数据安全分类分级标准指南",
) -> tuple[CanonicalStandard, StandardBuildReport]:
    """Build the finance canonical standard from raw guide entries."""
    report = StandardBuildReport(
        dataset=dataset,
        id_strategy="path",
        standard_name=standard_name,
        source_file=source_file,
        source_sheet=source_sheet,
        entries_read=len(entries),
    )
    categories: list[StandardCategory] = []
    for entry in entries:
        level_1 = clean(entry.level_1 if hasattr(entry, "level_1") else entry.get("level_1"))
        level_2 = clean(entry.level_2 if hasattr(entry, "level_2") else entry.get("level_2"))
        level_3 = clean(entry.level_3 if hasattr(entry, "level_3") else entry.get("level_3"))
        leaf = clean(entry.leaf if hasattr(entry, "leaf") else entry.get("leaf"))
        content = clean(entry.description if hasattr(entry, "description") else entry.get("description"))
        raw_level = str(entry.raw_level if hasattr(entry, "raw_level") else entry.get("raw_level", "")).strip()
        row = entry.row if hasattr(entry, "row") else entry.get("row")

        if not leaf:
            _add_issue(report, "empty_leaf_skipped", f"row {row}: empty leaf")
            continue
        # identity: dataset identity_fields = (level_1, level_2, level_4); a
        # present 三级子类 stays provenance-only (no invented level_3 slot)
        category_id = qualified_category_id(dataset, (level_1, level_2, leaf))
        path = _dedupe_path((level_1, level_2, level_3, leaf))
        level, raw_clean = normalize_standard_level(raw_level)
        if raw_level and level is None:
            _add_issue(
                report,
                "level_unparseable",
                f"row {row} category {leaf!r}: raw level {raw_clean!r} kept as-is, "
                f"standard_data_level=null (not guessed)",
            )
        categories.append(
            StandardCategoryBuilder(
                category_id=category_id,
                name=leaf,
                path=path,
                description=content,
                code=None,
                raw_level=raw_level,
                source_file=source_file,
                source_sheet=source_sheet,
                source_row=row,
            ).build()
        )
    report.categories_out = len(categories)
    final = _aggregate(categories, report)
    _finalize_report_levels(report, final)
    return CanonicalStandard(
        dataset=dataset,
        id_strategy="path",
        standard_source=SourceRef(file=source_file, sheet=source_sheet),
        standard_name=standard_name,
        categories=tuple(final),
    ), report


def build_shougang_standard(
    entries: Sequence[Any],
    *,
    source_file: str,
    source_sheet: str,
    dataset: str = "shougang",
    standard_name: str = "首钢京唐数据分类分级目录（关基）",
) -> tuple[CanonicalStandard, StandardBuildReport]:
    """Build the shougang canonical standard from the raw guanji catalog."""
    report = StandardBuildReport(
        dataset=dataset,
        id_strategy="code",
        standard_name=standard_name,
        source_file=source_file,
        source_sheet=source_sheet,
        entries_read=len(entries),
    )
    categories: list[StandardCategory] = []
    for entry in entries:
        level_1 = clean(entry.level_1 if hasattr(entry, "level_1") else entry.get("level_1"))
        level_2 = clean(entry.level_2 if hasattr(entry, "level_2") else entry.get("level_2"))
        level_3 = clean(entry.level_3 if hasattr(entry, "level_3") else entry.get("level_3"))
        raw_leaf = clean(entry.leaf if hasattr(entry, "leaf") else entry.get("leaf"))
        description = clean(entry.description if hasattr(entry, "description") else entry.get("description"))
        content = clean(entry.content if hasattr(entry, "content") else entry.get("content"))
        raw_level = str(entry.raw_level if hasattr(entry, "raw_level") else entry.get("raw_level", "")).strip()
        row = entry.row if hasattr(entry, "row") else entry.get("row")

        # a "——"/empty leaf cell means the catalog's leaf lives one level up:
        # fall back to the 三级 cell (the real reader already resolves this, but
        # accept dict fixtures and enforce the invariant here too)
        if not raw_leaf or raw_leaf.lower() in _PLACEHOLDER_NAMES or raw_leaf in _PLACEHOLDER_NAMES:
            raw_leaf = clean(entry.level_3 if hasattr(entry, "level_3") else entry.get("level_3"))
            if not raw_leaf or raw_leaf in _PLACEHOLDER_NAMES:
                _add_issue(
                    report,
                    "placeholder_skipped",
                    f"row {row}: no real leaf level (all ——); skipped and reported",
                )
                continue
        match = _TRAILING_CODE_RE.search(raw_leaf)
        if match:
            code = match.group(1)
            name = strip_code(raw_leaf)
        else:
            _add_issue(
                report,
                "no_code",
                f"row {row}: leaf {raw_leaf!r} has no category code; skipped "
                f"(identity is code-based, not guessed)",
            )
            continue
        path = _dedupe_path((strip_code(level_1), strip_code(level_2), strip_code(level_3), name))
        level, raw_clean = normalize_standard_level(raw_level)
        if raw_level and level is None:
            _add_issue(
                report,
                "level_unparseable",
                f"row {row} category {name!r}: raw level {raw_clean!r} kept as-is, "
                f"standard_data_level=null (not guessed)",
            )
        categories.append(
            StandardCategoryBuilder(
                category_id=code,
                name=name,
                path=path,
                description=description,
                code=code,
                raw_level=raw_level,
                content=content,
                source_file=source_file,
                source_sheet=source_sheet,
                source_row=row,
            ).build()
        )
    report.categories_out = len(categories)
    final = _aggregate(categories, report)
    _finalize_report_levels(report, final)
    return CanonicalStandard(
        dataset=dataset,
        id_strategy="code",
        standard_source=SourceRef(file=source_file, sheet=source_sheet),
        standard_name=standard_name,
        categories=tuple(final),
    ), report


def resolve_standard_dataset(dataset: str) -> str | None:
    """Which canonical standard owns a dataset's category facts.

    - finance -> finance
    - shougang -> shougang
    - infra   -> shougang (reuses the shared guanji standard; no copy)
    - pers_info -> None (no confirmed classification/grading standard)
    """
    return {
        "finance": "finance",
        "shougang": "shougang",
        "infra": "shougang",
        "pers_info": None,
    }[dataset]


__all__ = [
    "BuildIssue",
    "StandardBuildReport",
    "build_finance_standard",
    "build_shougang_standard",
    "resolve_standard_dataset",
]
