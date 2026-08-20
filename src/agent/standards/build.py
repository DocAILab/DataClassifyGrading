"""Canonical standard builders (Phase 1).

Turn raw standard rows (from merge-aware ``sources``) into a LOSSESS,
auditable ``CanonicalStandard``:
- ONE entry per real standard row — no aggregation in the fact layer.
  ``standard_entry_id`` is the true source identity; ``category_id`` is the
  legacy training/registry alias (projection; Phase 2 decides membership).
- Hierarchy facts that a leaf INHERITS from a merged group (finance 二级/三级
  定义, shougang 一级/二级/三级 定义, resource) are kept per entry in
  ``raw_fields`` WITH their source-cell / merged-range provenance.
- GRID-scoped annotations (finance 备注 J, 部门意见 K) are kept at standard
  level as ``ScopedAnnotation`` carrying their original merged range and the
  entry ids they apply to — never copied into a single leaf as private info.
- Grading columns are normalized L1..L4; unparseable values are reported and
  kept raw, never fixed. Placeholder / malformed rows (shougang \"——\" with no
  code, NaN) are skipped and reported; reader-level issues are merged in.

Deterministic: every entry preserved and sorted by standard_entry_id;
annotations sorted by annotation_id; no reliance on first-seen order.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from agent.task.identity import qualified_category_id
from agent.standards.contracts import (
    CanonicalStandard,
    ScopedAnnotation,
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


def _entry_value(entry: Any, key: str) -> Any:
    if hasattr(entry, key):
        return getattr(entry, key)
    if isinstance(entry, Mapping):
        return entry.get(key)
    return None


def _prov(entry: Any, key: str) -> Mapping[str, Any]:
    provenance = _entry_value(entry, "provenance") or {}
    value = provenance.get(key)
    return value if isinstance(value, Mapping) else {}


def _raw_field(entry: Any, key: str):
    """Build one raw_fields item ``{value, source_cell, merged_range, …}`` for
    a non-empty inherited hierarchy field, else None."""
    value = clean(_entry_value(entry, key) or "")
    if not value:
        return None
    info = dict(_prov(entry, key))
    info.pop("value", None)
    item = {"value": value}
    item.update(info)
    return item


def _scoped_annotations(
    dataset: str,
    rows: Sequence[tuple[int, str, str, str, str, str, int, int]],
) -> tuple[ScopedAnnotation, ...]:
    """Build gold-scope annotations from per-entry annotation sightings.

    Groups sightings by ``(type, text, scope_key)`` where scope_key is the
    merged range ("J93:J132") — or the cell itself ("J55") for an unmerged
    cell. Two unmerged cells with identical text are therefore DIFFERENT
    annotations (each keeps its own row), never merged into one spanning
    range. ``start_row/end_row`` come from the merged-range provenance or the
    single cell's own row, never from the observed member subset.
    """
    groups: dict[tuple[str, str, str], list[tuple[int, str, str, int, int]]] = defaultdict(list)
    for row, entry_id, type_, text, source_cell, merged_range, start, end in rows:
        if not text:
            continue
        scope_key = merged_range or source_cell or f"cell-{row}"
        groups[(type_, text, scope_key)].append((row, entry_id, source_cell, start, end))

    annotations: list[ScopedAnnotation] = []
    for (type_, text, scope_key), members in groups.items():
        members.sort(key=lambda m: (m[3] if m[3] is not None else m[0], m[0]))
        start_rows = [m[3] if m[3] is not None else m[0] for m in members]
        end_rows = [m[4] if m[4] is not None else m[0] for m in members]
        start_row = min(start_rows)
        end_row = max(end_rows)
        source_cell = members[0][2]
        # a merged range always contains ':' (e.g. "J93:J132"); a plain cell
        # ("J55") is not a range -> merged_range=None keeps the single scope
        merged_range = scope_key if ":" in scope_key else None
        annotation_id = f"{dataset}-{type_}-{start_row}-{end_row}"
        annotations.append(
            ScopedAnnotation(
                annotation_id=annotation_id,
                type=type_,
                text=text,
                source_cell=source_cell or scope_key,
                merged_range=merged_range,
                start_row=start_row,
                end_row=end_row,
                applies_to_standard_entry_ids=tuple(
                    sorted(entry_id for _, entry_id, _, _, _ in members)
                ),
            )
        )
    ids = [a.annotation_id for a in annotations]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate annotation ids (internal grouping error): {ids}")
    return tuple(sorted(annotations, key=lambda a: a.annotation_id))


def _finalize_report(
    report: StandardBuildReport,
    entries: Sequence[StandardCategory],
    reader_issues: Sequence[str],
) -> None:
    level_counter: Counter[str] = Counter()
    for entry in entries:
        level_counter[entry.standard_data_level or "''"] += 1
    report.level_distribution = dict(sorted(level_counter.items()))
    report.training_categories = len({entry.category_id for entry in entries})
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
    """Build the finance canonical standard from merge-aware raw entries.

    standard_entry_id = finance:L1.L2.L3.leaf (true identity incl. 三级);
    category_id = finance:L1.L2.leaf (training alias). Inherited 二级/三级
    定义 go to ``raw_fields``; 备注/部门意见 become ``scoped_annotations``.
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
    annotation_rows: list[tuple[int, str, str, str, str, str, int, int]] = []
    for entry in entries:
        level_1 = clean(_entry_value(entry, "level_1") or "")
        level_2 = clean(_entry_value(entry, "level_2") or "")
        level_3 = clean(_entry_value(entry, "level_3") or "")
        leaf = clean(_entry_value(entry, "leaf") or "")
        content = clean(_entry_value(entry, "description") or "")
        raw_level = str(_entry_value(entry, "raw_level") or "").strip()
        row = _entry_value(entry, "row")

        if not leaf:
            _add_issue(report, "empty_leaf_skipped", f"row {row}: empty leaf")
            continue
        standard_entry_id = qualified_category_id(dataset, (level_1, level_2, level_3, leaf))
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
        raw_fields: dict[str, Any] = {}
        for key in ("level_2_definition", "level_3_definition"):
            item = _raw_field(entry, key)
            if item is not None:
                raw_fields[key] = item

        built.append(
            StandardCategoryBuilder(
                standard_entry_id=standard_entry_id,
                category_id=category_id,
                name=leaf,
                path=path,
                description=content,
                code=None,
                raw_level=raw_level,
                raw_fields=raw_fields,
                source_file=source_file,
                source_sheet=source_sheet,
                source_row=row,
            ).build()
        )
        # scoped-annotation sightings (remark / department opinion)
        remark = clean(_entry_value(entry, "remark") or "")
        opinion = clean(_entry_value(entry, "department_opinion") or "")
        if remark:
            _push_annotation_sighting(
                annotation_rows, row, standard_entry_id, "remark", remark,
                _prov(entry, "remark"),
            )
        if opinion:
            _push_annotation_sighting(
                annotation_rows, row, standard_entry_id, "department_opinion",
                opinion, _prov(entry, "department_opinion"),
            )

    built.sort(key=lambda c: c.standard_entry_id)
    _assert_unique_entry_ids(built, "finance")
    report.standard_entries_out = len(built)
    _finalize_report(report, built, reader_issues)
    scoped = _scoped_annotations(dataset, annotation_rows)
    return CanonicalStandard(
        dataset=dataset,
        id_strategy="path",
        standard_source=SourceRef(file=source_file, sheet=source_sheet),
        standard_name=standard_name,
        entries=tuple(built),
        scoped_annotations=scoped,
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

    standard_entry_id == category_id == guanji code. Inherited 一级/二级/三级
    定义 and 数据来源(resource) are kept in ``raw_fields`` with provenance.
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
        level_1 = clean(_entry_value(entry, "level_1") or "")
        level_2 = clean(_entry_value(entry, "level_2") or "")
        level_3 = clean(_entry_value(entry, "level_3") or "")
        raw_leaf = clean(_entry_value(entry, "leaf") or "")
        description = clean(_entry_value(entry, "description") or "")
        content = clean(_entry_value(entry, "content") or "")
        raw_level = str(_entry_value(entry, "raw_level") or "").strip()
        row = _entry_value(entry, "row")

        if not raw_leaf or raw_leaf.lower() in _PLACEHOLDER_NAMES or raw_leaf in _PLACEHOLDER_NAMES:
            raw_leaf = clean(_entry_value(entry, "level_3") or "")
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
        raw_fields: dict[str, Any] = {}
        for key in ("level_1_definition", "level_2_definition", "level_3_definition", "resource"):
            item = _raw_field(entry, key)
            if item is not None:
                raw_fields[key] = item

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
                raw_fields=raw_fields,
                source_file=source_file,
                source_sheet=source_sheet,
                source_row=row,
            ).build()
        )
    built.sort(key=lambda c: c.standard_entry_id)
    _assert_unique_entry_ids(built, "shougang")
    report.standard_entries_out = len(built)
    _finalize_report(report, built, reader_issues)
    return CanonicalStandard(
        dataset=dataset,
        id_strategy="code",
        standard_source=SourceRef(file=source_file, sheet=source_sheet),
        standard_name=standard_name,
        entries=tuple(built),
    ), report


def _push_annotation_sighting(
    rows: list[tuple[int, str, str, str, str, str, int, int]],
    row: int,
    entry_id: str,
    type_: str,
    text: str,
    prov: Mapping[str, Any],
) -> None:
    start = prov.get("start_row")
    end = prov.get("end_row")
    rows.append(
        (
            row,
            entry_id,
            type_,
            text,
            str(prov.get("source_cell") or ""),
            prov.get("merged_range"),
            int(start) if start is not None else None,
            int(end) if end is not None else None,
        )
    )


def _assert_unique_entry_ids(built: Sequence[StandardCategory], dataset: str) -> None:
    unique = {entry.standard_entry_id for entry in built}
    if len(unique) != len(built):
        raise ValueError(
            f"{dataset} standard_entry_id must be unique; got "
            f"{len(built) - len(unique)} duplicate(s)"
        )


def resolve_standard_dataset(dataset: str) -> str | None:
    """Which canonical standard owns a dataset's category facts."""
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
