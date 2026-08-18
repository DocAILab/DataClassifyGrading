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

from agent.task.contracts import CorpusCategory
from agent.task.dataset_config import BUILTIN_DATASET_CONFIGS, DatasetConfig
from agent.task.identity import leaf_registry_from_corpus, qualified_category_id

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
    """Per-dataset conversion report (skips, aggregations, coverage gaps)."""

    dataset: str
    source: str | None
    id_strategy: str
    entries_read: int = 0
    categories_out: int = 0
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
    registry_ids: set[str],
    records: Sequence[Mapping[str, Any]],
    config: DatasetConfig,
    code_map: Mapping[str, str],
) -> dict[str, Any]:
    """How many dataset records resolve to an ID present in the registry.

    Uses the stage-2 resolver semantics (code or path strategy). This is a
    diagnostic only — it does not modify the registry or the records.
    """
    from agent.task.resolver import ClassificationTargetResolver

    resolver = ClassificationTargetResolver(config, code_leaf_map=code_map)
    covered = 0
    unresolved = 0
    skipped = 0
    for record in records:
        target = resolver.resolve(record)
        if target is None:
            skipped += 1
        elif target.category_id in registry_ids:
            covered += 1
        else:
            unresolved += 1
    return {
        "records": len(records),
        "covered": covered,
        "uncovered": unresolved,
        "skipped": skipped,
        "coverage_rate": round(covered / len(records), 4) if records else 0.0,
    }


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
    "compute_dataset_id_coverage",
]
