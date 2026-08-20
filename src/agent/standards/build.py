"""Canonical standard builders (Phase 1).

Turn raw standard rows (from ``sources``) into a LOSSESS, auditable
``CanonicalStandard``:
- ONE entry per real standard row — there is NO aggregation in the fact layer.
  ``standard_entry_id`` is the true source identity; ``category_id`` is the
  legacy training/registry alias (a projection, possibly shared by several
  entries). The 237 finance rows therefore stay 237 entries; the 237→233
  ``training_projection`` is exposed as a DERIVED view for Phase 2.
- category_id continues the existing stable identity strategy (finance L1-L2-leaf
  via DatasetConfig.identity_fields; shougang guanji code).
- path stores the TRUE source-hierarchy depth (empty 三级子类 omitted — no
  invented padding).
- grading columns are normalized to L1..L4 with ``normalize_standard_level``;
  unparseable values are reported and kept raw, never fixed or guessed.
- placeholder / malformed rows (shougang "——", NaN, missing code) are skipped
  and reported in the build report — never silently repaired. Reader-level
  issues are merged into the same report.

Deterministic for identical input regardless of the order entries arrive in
(every entry is preserved and sorted by standard_entry_id; nothing depends on
"first seen"). The CLI writes every artifact only after all datasets build and
align successfully (fail-fast) and refuses to overwrite without --overwrite.
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
    standard_entries_out: int = 0
    training_categories: int = 0
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
            "standard_entries_out": self.standard_entries_out,
            "training_categories": self.training_categories,
            "level_distribution": dict(self.level_distribution),
            "issues": sorted(
                (issue.to_mapping() for issue in self.issues),
                key=lambda i: (i["kind"], i["detail"]),
            ),
        }


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


def _finalize_report(
    report: StandardBuildReport,
    entries: Sequence[StandardCategory],
    reader_issues: Sequence[str],
) -> None:
    """Level distribution + projection counts + merged reader issues.

    Deterministic: issues are sorted before persistence (see to_mapping).
    """
    level_counter: Counter[str] = Counter()
    for entry in entries:
        level_counter[entry.standard_data_level or "''"] += 1
    report.level_distribution = dict(sorted(level_counter.items()))
    report.training_categories = len(
        {entry.category_id for entry in entries}
    )
    for detail in reader_issues:
        report.issues.append(BuildIssue(kind="reader_issue", detail=str(detail)))


def build_finance_standard(
    entries: Sequence[Any],
    *,
    source_file: str,
    source_sheet: str,
    dataset: str = "finance",
    standard_name: str = "金融行业数据安全分类分级标准指南",
    reader_issues: Sequence[str] = (),
) -> tuple[CanonicalStandard, StandardBuildReport]:
    """Build the finance canonical standard from raw guide entries.

    One entry per raw row. ``standard_entry_id`` = L1.L2.L3.leaf (the TRUE
    source identity, level_3 included); ``category_id`` = L1.L2.leaf (the
    training alias, matching DatasetConfig.identity_fields). Several entries
    may share a category_id (e.g. 业务/合约协议/基本信息 under five 三级):
    they stay separate entries, exposed via ``training_projection``.
    """
    report = StandardBuildReport(
        dataset=dataset,
        id_strategy="path",
        standard_name=standard_name,
        source_file=source_file,
        source_sheet=source_sheet,
        entries_read=len(entries),
    )
    built: list[StandardCategory] = []
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
        # TRUE identity keeps the real 三级子类 (empty slots stay empty parts)
        standard_entry_id = qualified_category_id(
            dataset, (level_1, level_2, level_3, leaf)
        )
        # training alias: DatasetConfig.identity_fields = (level_1, level_2, level_4)
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
        built.append(
            StandardCategoryBuilder(
                standard_entry_id=standard_entry_id,
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
    # lossless fact layer: every entry preserved; determinism by entry-id order
    built.sort(key=lambda c: c.standard_entry_id)
    duplicate_ids = {e.standard_entry_id for e in built}
    if len(duplicate_ids) != len(built):
        raise ValueError(
            "finance standard_entry_id must be unique; got "
            f"{len(built) - len(duplicate_ids)} duplicate(s)"
        )
    report.standard_entries_out = len(built)
    _finalize_report(report, built, reader_issues)
    return CanonicalStandard(
        dataset=dataset,
        id_strategy="path",
        standard_source=SourceRef(file=source_file, sheet=source_sheet),
        standard_name=standard_name,
        entries=tuple(built),
    ), report


def build_shougang_standard(
    entries: Sequence[Any],
    *,
    source_file: str,
    source_sheet: str,
    dataset: str = "shougang",
    standard_name: str = "首钢京唐数据分类分级目录（关基）",
    reader_issues: Sequence[str] = (),
) -> tuple[CanonicalStandard, StandardBuildReport]:
    """Build the shougang canonical standard from the raw guanji catalog.

    standard_entry_id == category_id == guanji code (opaque identity). A
    \"——\"/empty leaf cell means the leaf lives one level up (三级 carries the
    real code/name); those are real categories resolved and kept, never lost.
    """
    report = StandardBuildReport(
        dataset=dataset,
        id_strategy="code",
        standard_name=standard_name,
        source_file=source_file,
        source_sheet=source_sheet,
        entries_read=len(entries),
    )
    built: list[StandardCategory] = []
    for entry in entries:
        level_1 = clean(entry.level_1 if hasattr(entry, "level_1") else entry.get("level_1"))
        level_2 = clean(entry.level_2 if hasattr(entry, "level_2") else entry.get("level_2"))
        level_3 = clean(entry.level_3 if hasattr(entry, "level_3") else entry.get("level_3"))
        raw_leaf = clean(entry.leaf if hasattr(entry, "leaf") else entry.get("leaf"))
        description = clean(entry.description if hasattr(entry, "description") else entry.get("description"))
        content = clean(entry.content if hasattr(entry, "content") else entry.get("content"))
        raw_level = str(entry.raw_level if hasattr(entry, "raw_level") else entry.get("raw_level", "")).strip()
        row = entry.row if hasattr(entry, "row") else entry.get("row")

        # a "——"/empty leaf cell means the leaf lives one level up (三级)
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
        built.append(
            StandardCategoryBuilder(
                standard_entry_id=code,
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
    built.sort(key=lambda c: c.standard_entry_id)
    duplicate_ids = {e.standard_entry_id for e in built}
    if len(duplicate_ids) != len(built):
        raise ValueError(
            "shougang standard_entry_id (code) must be unique; got "
            f"{len(built) - len(duplicate_ids)} duplicate(s)"
        )
    report.standard_entries_out = len(built)
    _finalize_report(report, built, reader_issues)
    return CanonicalStandard(
        dataset=dataset,
        id_strategy="code",
        standard_source=SourceRef(file=source_file, sheet=source_sheet),
        standard_name=standard_name,
        entries=tuple(built),
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
