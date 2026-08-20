"""Formal conversion: standard/corpus -> CorpusCategory -> LeafRegistry.

Pipeline per dataset (stage 3A):

    raw standard (standards_map JSON)
        -> parse_*_standard()          one CorpusCategory per standard entry
        -> aggregate_categories()      same category_id -> examples aggregation
        -> canonical corpus JSON       (category_id/name/description/path/code/examples)
        -> leaf_registry_from_corpus() -> registry JSON (LeafRegistry format)

ID strategy follows the stage-2 contract:
- shougang / infra (guanji_dict): category_id = guanji code (opaque identity;
  digit groups are not interpreted as levels). Malformed entries (the
  'NaN' key with category 'nan') are skipped and reported, never repaired.
- finance (financial_standards_dict): the standard is a 3-segment path
  (L1-L2-leaf) and carries no code; category_id is the deterministic
  path-qualified ID built from the canonical standard path itself (no
  empty level_3 slot is invented). The dataset's level_3 is provenance
  only and never participates in the finance category identity; the
  resolver maps dataset L1/L2/L4 onto the canonical L1/L2/L4 slots via
  DatasetConfig.identity_fields.
- pers_info: no standard/corpus exists (UNKNOWN); the 18-category registry
  is derived from the complete dataset universe (all.json leaf space, never
  a train/val/test split) and is explicitly marked derived_from
  "dataset-universe" until a real corpus is confirmed.

No semantic matching, no label fixing; unknowns stay unresolved and are
listed in the build report. Artifact sources are repo-relative logical
paths (no machine-local absolute paths).
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from agent.task.contracts import CorpusCategory, CorpusScopedAnnotation, StandardEntryView
from agent.task.dataset_config import BUILTIN_DATASET_CONFIGS, DatasetConfig
from agent.task.identity import compact, leaf_registry_from_corpus, qualified_category_id

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_TRAILING_CODE_RE = re.compile(
    r"\s*[\(\[（【]\s*([A-Za-z]+\d*(?:-\d+)*)\s*[\)\]）】]\s*$"
)
_MALFORMED_NAMES = {"nan", ""}


@dataclass
class ParseIssue:
    """One data-quality issue found while parsing a standard (not repaired)."""

    kind: str
    detail: str

    def to_mapping(self) -> dict[str, str]:
        return {"kind": self.kind, "detail": self.detail}


@dataclass
class BuildReport:
    """Per-dataset conversion report (skips, aggregations, coverage gaps).

    Phase-2 fields: ``standard_entries_out`` (registry/corpus derived from the
    234/237 canonical standard entries before projection), ``excluded_categories``
    (standard categories NOT activated because they were absent from the
    previous active registry — e.g. shougang B3-6 — reported, not silently
    added or dropped).
    """

    dataset: str
    source: str | None
    id_strategy: str
    entries_read: int = 0
    categories_out: int = 0
    standard_entries_out: int = 0
    excluded_categories: list[str] = field(default_factory=list)
    standard_name: str = ""
    issues: list[ParseIssue] = field(default_factory=list)
    aggregated: dict[str, int] = field(default_factory=dict)
    registry_source: str | None = None
    dataset_id_coverage: dict[str, Any] = field(default_factory=dict)

    def to_mapping(self) -> dict[str, Any]:
        mapping: dict[str, Any] = {
            "dataset": self.dataset,
            "source": self.source,
            "id_strategy": self.id_strategy,
            "entries_read": self.entries_read,
            "categories_out": self.categories_out,
            "standard_entries_out": self.standard_entries_out,
            "excluded_categories": list(self.excluded_categories),
            "standard_name": self.standard_name,
            "aggregated": dict(self.aggregated),
            "issues": [issue.to_mapping() for issue in self.issues],
        }
        if self.registry_source is not None:
            mapping["registry_source"] = self.registry_source
        mapping["dataset_id_coverage"] = self.dataset_id_coverage
        return mapping


def _repo_relative(path: str | Path) -> str:
    """Repo-relative logical path (POSIX separators); falls back to the raw
    path when the file lives outside the repository."""
    try:
        return Path(path).resolve().relative_to(_PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return Path(path).as_posix()


def _clean(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value).strip())


def _category_id_slots(dataset: str, slots: Sequence[str]) -> str:
    """Path-qualified ID over fixed path_fields slots (empty slots kept)."""
    return qualified_category_id(dataset, slots)


# ---------------------------------------------------------------------------
# standard parsers
# ---------------------------------------------------------------------------


def parse_guanji_standard(
    mapping: Mapping[str, Any],
    dataset: str = "shougang",
) -> tuple[list[CorpusCategory], list[ParseIssue]]:
    """Parse guanji_dict: ``name（A1-1-1）`` entries keyed by description.

    category_id = code. The 'NaN'/'nan' malformed entry is skipped and
    reported (never repaired or silently dropped without a trace).
    """
    categories: list[CorpusCategory] = []
    issues: list[ParseIssue] = []
    for key, value in mapping.items():
        raw = value.get("category") if isinstance(value, Mapping) else value
        text = _clean(raw)
        if not text or text.lower() in _MALFORMED_NAMES:
            issues.append(
                ParseIssue(
                    "malformed_skipped",
                    f"key={key!r} category={text!r}",
                )
            )
            continue
        match = _TRAILING_CODE_RE.search(text)
        if match:
            code = match.group(1)
            name = text[: match.start()].strip()
        else:
            code = None
            name = text
        category_id = code if code else _category_id_slots(dataset, (name,))
        if code is None:
            issues.append(
                ParseIssue(
                    "no_code",
                    f"category {name!r} has no code; id fell back to path "
                    f"slot {category_id!r}",
                )
            )
        categories.append(
            CorpusCategory(
                category_id=category_id,
                name=name,
                description=_clean(key),
                code=code,
            )
        )
    return categories, issues


def parse_financial_standard(
    mapping: Mapping[str, Any],
    dataset: str = "finance",
) -> tuple[list[CorpusCategory], list[ParseIssue]]:
    """Parse financial_standards_dict: ``L1-L2-leaf`` dash paths, no codes.

    category_id is built from the canonical standard path itself (L1-L2-leaf,
    three slots, no invented empty level_3); the category path keeps the
    original segments as provenance. Entries whose level_1 is outside the
    dataset's vocabulary (e.g. '监管-*') are still part of the standard
    universe and kept.
    """
    categories: list[CorpusCategory] = []
    issues: list[ParseIssue] = []
    for key, value in mapping.items():
        raw = value.get("category") if isinstance(value, Mapping) else value
        text = _clean(raw)
        if not text:
            issues.append(ParseIssue("malformed_skipped", f"key={key!r} empty category"))
            continue
        parts = [part.strip() for part in text.split("-")]
        if len(parts) >= 2 and all(parts):
            leaf = parts[-1]
            slots = tuple(parts)
            path = tuple(parts)
        else:
            issues.append(
                ParseIssue(
                    "unexpected_format",
                    f"category {text!r} is not a dash path; kept as single leaf",
                )
            )
            leaf = text
            slots = (leaf,)
            path = ()
        categories.append(
            CorpusCategory(
                category_id=_category_id_slots(dataset, slots),
                name=leaf,
                description=_clean(key),
                path=path,
                code=None,
            )
        )
    return categories, issues


def pers_info_categories(
    records: Iterable[Mapping[str, Any]],
    dataset: str = "pers_info",
) -> tuple[list[CorpusCategory], list[ParseIssue]]:
    """Build the pers_info 18-category universe from all.json leaf values.

    No standard/corpus exists (stage-1 report: corpus covers 4/18 leaves).
    The universe is the complete dataset leaf space (all records, never a
    train/val/test split); descriptions are missing by design. The output is
    marked derived_from 'dataset-universe' until a real corpus is confirmed.
    """
    leaves: set[str] = set()
    for record in records:
        classification = record.get("classification")
        if not isinstance(classification, Mapping):
            continue
        leaf = _clean(classification.get("level_4"))
        if leaf:
            leaves.add(leaf)
    categories = [
        CorpusCategory(
            category_id=_category_id_slots(dataset, (leaf,)),
            name=leaf,
            description="",
            path=(leaf,),
            code=None,
        )
        for leaf in sorted(leaves)
    ]
    return categories, []


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------


def aggregate_categories(
    categories: Iterable[CorpusCategory],
) -> tuple[list[CorpusCategory], dict[str, int]]:
    """Merge entries with the same category_id.

    The first entry keeps its primary ``description``; every further entry's
    description is appended to ``descriptions`` (a category may own several
    guide documents describing it). ``examples`` is never used for
    descriptions: descriptions and examples stay semantically distinct.
    Returns (categories, {"kinds": n, "instances": m}) where kinds = number
    of category_ids that occurred more than once and instances = total
    number of extra (merged-away) entries.
    """
    counts: Counter[str] = Counter()
    by_id: dict[str, CorpusCategory] = {}
    for category in categories:
        counts[category.category_id] += 1
        existing = by_id.get(category.category_id)
        if existing is None:
            by_id[category.category_id] = category
            continue
        if category.description:
            existing = CorpusCategory(
                category_id=existing.category_id,
                name=existing.name,
                description=existing.description,
                descriptions=existing.descriptions + (category.description,),
                path=existing.path,
                code=existing.code,
                examples=existing.examples,
            )
            by_id[category.category_id] = existing
    merged = {category_id: count for category_id, count in counts.items() if count > 1}
    aggregated = {
        "kinds": len(merged),
        "instances": sum(count - 1 for count in merged.values()),
    }
    return list(by_id.values()), aggregated


# ---------------------------------------------------------------------------
# JSON mapping
# ---------------------------------------------------------------------------


def corpus_category_to_mapping(category: CorpusCategory) -> dict[str, Any]:
    mapping: dict[str, Any] = {
        "category_id": category.category_id,
        "name": category.name,
        "description": category.description,
        "path": list(category.path),
        "code": category.code,
    }
    if category.descriptions:
        mapping["descriptions"] = list(category.descriptions)
    if category.examples:
        mapping["examples"] = list(category.examples)
    if category.standard_entry_ids:
        mapping["standard_entry_ids"] = list(category.standard_entry_ids)
    if category.standard_entries:
        mapping["standard_entries"] = [
            entry.to_mapping() for entry in category.standard_entries
        ]
    if category.scoped_annotations:
        mapping["scoped_annotations"] = [
            annotation.to_mapping() for annotation in category.scoped_annotations
        ]
    return mapping


def corpus_to_mapping(
    dataset: str,
    source: str | None,
    id_strategy: str,
    categories: Sequence[CorpusCategory],
    report: BuildReport,
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "source": source,
        "id_strategy": id_strategy,
        "categories": [corpus_category_to_mapping(c) for c in categories],
        "build_report": report.to_mapping(),
    }


def registry_to_mapping(categories: Sequence[CorpusCategory]) -> dict[str, Any]:
    """LeafRegistry.from_mapping-compatible JSON (no examples field)."""
    return {
        "categories": [
            {
                "category_id": c.category_id,
                "name": c.name,
                "description": c.description,
                "path": list(c.path),
                "code": c.code,
            }
            for c in categories
        ]
    }


# ---------------------------------------------------------------------------
# per-dataset builders
# ---------------------------------------------------------------------------


def build_shougang_corpus(
    standard_path: str | Path,
    dataset: str = "shougang",
) -> tuple[list[CorpusCategory], BuildReport]:
    with Path(standard_path).open(encoding="utf-8") as handle:
        standard = json.load(handle)
    categories, issues = parse_guanji_standard(standard, dataset=dataset)
    categories, aggregated = aggregate_categories(categories)
    report = BuildReport(
        dataset=dataset,
        source=_repo_relative(standard_path),
        id_strategy="code",
        entries_read=len(standard),
        categories_out=len(categories),
        issues=issues,
        aggregated=aggregated,
    )
    return categories, report


def build_infra_corpus(
    standard_path: str | Path,
    dataset: str = "infra",
) -> tuple[list[CorpusCategory], BuildReport]:
    """Infra shares the shougang category universe (guanji) but keeps its own
    build report and records the shared registry source explicitly."""
    with Path(standard_path).open(encoding="utf-8") as handle:
        standard = json.load(handle)
    categories, issues = parse_guanji_standard(standard, dataset=dataset)
    categories, aggregated = aggregate_categories(categories)
    report = BuildReport(
        dataset=dataset,
        source=_repo_relative(standard_path),
        id_strategy="code",
        entries_read=len(standard),
        categories_out=len(categories),
        issues=issues,
        aggregated=aggregated,
        registry_source="shougang",
    )
    return categories, report


def build_finance_corpus(
    standard_path: str | Path,
    dataset: str = "finance",
) -> tuple[list[CorpusCategory], BuildReport]:
    with Path(standard_path).open(encoding="utf-8") as handle:
        standard = json.load(handle)
    categories, issues = parse_financial_standard(standard, dataset=dataset)
    categories, aggregated = aggregate_categories(categories)
    report = BuildReport(
        dataset=dataset,
        source=_repo_relative(standard_path),
        id_strategy="path",
        entries_read=len(standard),
        categories_out=len(categories),
        issues=issues,
        aggregated=aggregated,
    )
    return categories, report


def build_pers_info_corpus(
    records: Iterable[Mapping[str, Any]],
    dataset: str = "pers_info",
) -> tuple[list[CorpusCategory], BuildReport]:
    categories, issues = pers_info_categories(records, dataset=dataset)
    entries_read = len(categories)
    categories, aggregated = aggregate_categories(categories)
    report = BuildReport(
        dataset=dataset,
        source=None,
        id_strategy="path",
        entries_read=entries_read,
        categories_out=len(categories),
        issues=issues,
        aggregated=aggregated,
    )
    report.issues.append(
        ParseIssue(
            "derived_from_dataset_universe",
            "no standard/corpus exists for pers_info (stage-1 report: corpus "
            "covers 4/18 leaves); the registry is derived from the complete "
            "all.json leaf space, not from any train split; replace when a "
            "real corpus is confirmed",
        )
    )
    return categories, report


# ---------------------------------------------------------------------------
# dataset id coverage (requires local data/, skipped when unavailable)
# ---------------------------------------------------------------------------


def compute_dataset_id_coverage(
    dataset: str,
    categories: Sequence[CorpusCategory],
    records: Sequence[Mapping[str, Any]],
    config: DatasetConfig,
    code_map: Mapping[str, str],
) -> dict[str, Any]:
    """How many dataset records resolve to an ID present in the registry.

    Uses the stage-2 resolver semantics (code or path strategy). Uncovered
    records are split into:
    - missing_leaf: the leaf name itself is absent from the corpus universe
      (e.g. the 5 finance leaves without standard definitions);
    - path_mismatch: a corpus-known leaf resolved under L1/L2 slots that
      differ from the standard path (e.g. finance '网络服务信息' under
      经营管理/技术管理 vs standard 经营管理/运营管理). These records are
      NOT auto-repaired: the discrepancy may be a dataset labeling error or
      a genuine standard/dataset hierarchy difference.
    This is a diagnostic only — it does not modify the registry or records.
    """
    from agent.task.resolver import ClassificationTargetResolver

    registry_ids = {category.category_id for category in categories}
    registry_names = {category.name for category in categories}
    resolver = ClassificationTargetResolver(config, code_leaf_map=code_map)
    covered = 0
    skipped = 0
    uncovered_by_id: Counter[str] = Counter()
    missing_leaf_by_name: Counter[str] = Counter()
    path_mismatch_by_id: Counter[str] = Counter()
    for record in records:
        target = resolver.resolve(record)
        if target is None:
            skipped += 1
        elif target.category_id in registry_ids:
            covered += 1
        else:
            uncovered_by_id[target.category_id] += 1
            if target.leaf_name in registry_names:
                path_mismatch_by_id[target.category_id] += 1
            else:
                missing_leaf_by_name[target.leaf_name] += 1
    return {
        "records": len(records),
        "covered": covered,
        "uncovered": sum(uncovered_by_id.values()),
        "skipped": skipped,
        "coverage_rate": round(covered / len(records), 4) if records else 0.0,
        "uncovered_ids": {key: value for key, value in sorted(uncovered_by_id.items())},
        "missing_leaf": {
            "count": sum(missing_leaf_by_name.values()),
            "leaves": {key: value for key, value in sorted(missing_leaf_by_name.items())},
        },
        "path_mismatch": {
            "count": sum(path_mismatch_by_id.values()),
            "ids": {key: value for key, value in sorted(path_mismatch_by_id.items())},
        },
    }


# ---------------------------------------------------------------------------
# Phase-2: derive registry/corpus from the canonical standard (lossless)
# ---------------------------------------------------------------------------


def _standard_registry_path(dataset: str, category_id: str) -> tuple[str, ...]:
    """Stage-1 registry path (kept byte-identical to the pre-Phase-2 values so
    Stage 1 prompt/choice behavior does not change):
    - finance: L1/L2/leaf (the training identity parts, no invented empty slot)
    - shougang/infra: empty (historical Stage-1 view; real path lives in the
      corpus ``standard_entries[].path``)
    """
    if dataset in ("shougang", "infra"):
        return ()
    if dataset == "finance":
        parts = category_id.split(":", 1)[-1].split(".")
        return tuple(part for part in parts if part)
    return ()


def _projection_exclusion(dataset: str) -> tuple[str, ...]:
    """Explicit projection policy from the dataset config.

    The ONLY formal-build input about universe membership: standard categories
    that must not enter the active registry/corpus (shougang B3-6). Datasets
    that reuse another registry (infra -> shougang) inherit that dataset's
    policy. The legacy registry is never read at build time (audit/parity only).
    """
    config = BUILTIN_DATASET_CONFIGS[dataset]
    policy_dataset = config.registry_source or dataset
    return BUILTIN_DATASET_CONFIGS[policy_dataset].projection_excluded_category_ids


def build_from_standard(
    standard,
    *,
    dataset: str = "finance",
    excluded_category_ids: set[str] | tuple[str, ...] | None = None,
    registry_source: str | None = None,
) -> tuple[list[CorpusCategory], BuildReport]:
    """Derive the LeafRegistry/Corpus universe from a Phase-1 CanonicalStandard.

    - ONE category per training alias (category_id), preserving Stage-1 fields
      (category_id/name/path/code + primary description) EXACTLY as the
      historical registry/corpus, ordered by first source row so Stage 1/2
      choice ids and prompts stay byte-identical.
    - LOSSESS per-entry facts are kept on the CorpusCategory (standard_entry_ids
      / standard_entries / scoped_annotations) — multiple standard entries under
      one category (e.g. finance 5× 基本信息) are never collapsed to "first".
    - ``excluded_category_ids``: the EXPLICIT projection policy (defaults to
      the dataset config's ``projection_excluded_category_ids``; infra inherits
      shougang's). Excluded standard categories (shougang B3-6) are removed
      from the active universe and REPORTED (excluded_categories +
      category_excluded issues) — never silently added or dropped.
    Returns (categories, report); nothing is written and no labels are touched.
    """
    from agent.standards.contracts import CanonicalStandard

    if not isinstance(standard, CanonicalStandard):
        raise TypeError("build_from_standard requires a CanonicalStandard")
    report = BuildReport(
        dataset=dataset,
        source=standard.standard_source.file or None,
        id_strategy=standard.id_strategy,
        entries_read=len(standard.entries),
        standard_name=standard.standard_name,
        registry_source=registry_source,
    )
    annotations_by_entry: dict[str, list[CorpusScopedAnnotation]] = {}
    for entry in standard.entries:
        annotations_by_entry[entry.standard_entry_id] = []
    for annotation in standard.scoped_annotations:
        for entry_id in annotation.applies_to_standard_entry_ids:
            annotations_by_entry.setdefault(entry_id, []).append(
                CorpusScopedAnnotation(
                    annotation_id=annotation.annotation_id,
                    type=annotation.type,
                    text=annotation.text,
                    source_cell=annotation.source_cell,
                    merged_range=annotation.merged_range,
                    start_row=annotation.start_row,
                    end_row=annotation.end_row,
                    applies_to_standard_entry_ids=annotation.applies_to_standard_entry_ids,
                )
            )

    # deterministic Excel-row order (the canonical standard itself sorts by
    # standard_entry_id, so re-sort by source row to mirror the historical
    # registry/corpus order)
    ordered = sorted(
        standard.entries, key=lambda e: (e.source.row if e.source.row is not None else 0, e.standard_entry_id)
    )
    aggregated: dict[str, list] = {}
    for entry in ordered:
        aggregated.setdefault(entry.category_id, []).append(entry)
    report.standard_entries_out = len(ordered)

    if excluded_category_ids is None:
        excluded_category_ids = _projection_exclusion(dataset)
    excluded_set = set(excluded_category_ids)
    active_ids = [
        category_id
        for category_id in aggregated
        if category_id not in excluded_set
    ]
    excluded = [
        category_id
        for category_id in aggregated
        if category_id in excluded_set
    ]
    report.excluded_categories = sorted(excluded)
    for category_id in sorted(excluded):
        report.issues.append(
            ParseIssue(
                "category_excluded",
                f"standard category {category_id!r} excluded by the explicit "
                "projection policy (DatasetConfig.projection_excluded_category_ids, "
                "e.g. shougang B3-6); not added to the active universe this "
                "phase — reported, not silently added or dropped",
            )
        )

    categories: list[CorpusCategory] = []
    for category_id in active_ids:
        entry_list = aggregated[category_id]
        first = entry_list[0]
        annotations = sorted(
            {
                annotation
                for entry in entry_list
                for annotation in annotations_by_entry.get(entry.standard_entry_id, [])
            },
            key=lambda a: a.annotation_id,
        )
        categories.append(
            CorpusCategory(
                category_id=category_id,
                # Stage-1/Stage-2-facing name: whitespace-compact, matching the
                # historical registry/corpus (the raw standard leaf may contain
                # a stray space, e.g. "个人基本概况 信息"); the source-faithful
                # per-entry name is kept in ``standard_entries[].name``.
                name=compact(first.name),
                # Stage-2-facing description keeps the SOURCE text (the
                # canonical standard is the fact source). One documented legacy
                # divergence: the old dict silently dropped a stray space in
                # the source cell of 经营管理/运营管理/档案资料管理信息
                # (whitespace-only, reported in the build; Stage-1 never uses
                # description).
                description=first.description,
                descriptions=tuple(entry.description for entry in entry_list[1:]),
                path=_standard_registry_path(dataset, category_id),
                code=first.code,
                standard_entry_ids=tuple(entry.standard_entry_id for entry in entry_list),
                standard_entries=tuple(
                    StandardEntryView(
                        standard_entry_id=entry.standard_entry_id,
                        name=entry.name,
                        path=entry.path,
                        description=entry.description,
                        raw_level=entry.raw_level,
                        standard_data_level=entry.standard_data_level,
                        content=entry.content,
                        code=entry.code,
                        source={
                            "file": entry.source.file,
                            "sheet": entry.source.sheet,
                            "row": entry.source.row,
                        },
                        raw_fields=dict(entry.raw_fields),
                    )
                    for entry in entry_list
                ),
                scoped_annotations=tuple(annotations),
            )
        )
    report.categories_out = len(categories)
    return categories, report


__all__ = [
    "ParseIssue",
    "BuildReport",
    "parse_guanji_standard",
    "parse_financial_standard",
    "pers_info_categories",
    "aggregate_categories",
    "corpus_category_to_mapping",
    "corpus_to_mapping",
    "registry_to_mapping",
    "build_shougang_corpus",
    "build_infra_corpus",
    "build_finance_corpus",
    "build_pers_info_corpus",
    "build_from_standard",
    "compute_dataset_id_coverage",
]
