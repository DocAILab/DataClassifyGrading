"""sample <-> canonical standard alignment audit (Phase 1).

Reads canonical records (data/canonical/<dataset>/all.json — the same records
the training pipeline consumes, unchanged) and joins them to a
``CanonicalStandard`` by the training alias ``category_id``.

Buckets (strict alias join):
- matched                     : sample level is among the entry levels of the category
- mismatched                  : entries have levels, sample level differs from all of them
- standard_level_unavailable  : the category's entries have no parseable level
- standard_missing           : sample category_id not in the standard
- sample_missing             : standard category observed in no resolved sample
- unresolved                 : canonical records without a resolved target, with
                               evidence-only candidate standard entries by leaf name

Pure computation: NEVER modifies sample data_level, never auto-repairs labels,
never guesses the meaning of a level. ``standard_data_level`` is only the
standard's category-level reference.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from agent.standards.contracts import CanonicalStandard


def load_canonical_records(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError(f"{path} must be a JSON list")
    return records


def align_dataset_to_standard(
    records: Sequence[Mapping[str, Any]],
    standard: CanonicalStandard,
    *,
    field_for_audit: str = "field_name",
) -> dict[str, Any]:
    """Return a deterministic alignment report (no mutation of ``records``)."""
    entries_by_category = standard.entries_by_category_id()
    counts: Counter[str] = Counter()
    unresolved_by_status: Counter[str] = Counter()
    mismatched: list[dict[str, Any]] = []
    standard_missing: list[dict[str, Any]] = []
    level_unavailable: list[dict[str, Any]] = []
    unresolved_evidence: list[dict[str, Any]] = []
    resolved_categories: set[str] = set()

    for record in records:
        counts["total"] += 1
        status = str(record.get("resolution_status", "") or "")
        target = record.get("target")
        if status != "resolved" or not isinstance(target, Mapping):
            unresolved_by_status[status or "(no status)"] += 1
            counts["unresolved"] += 1
            unresolved_evidence.append(
                _unresolved_evidence(record, status, standard)
            )
            continue
        category_id = str(target.get("category_id", "") or "")
        sample_level = str(record.get("data_level", "") or "")
        counts["resolved"] += 1
        resolved_categories.add(category_id)

        entries = entries_by_category.get(category_id)
        if not entries:
            counts["standard_missing"] += 1
            standard_missing.append(
                {
                    "category_id": category_id,
                    "name": str(target.get("leaf_name", "") or ""),
                    "sample_level": sample_level,
                    "path": list(target.get("category_path") or ()),
                    "field": _audit_field(record, field_for_audit),
                }
            )
            continue
        entry_levels = {
            entry.standard_data_level
            for entry in entries
            if entry.standard_data_level is not None
        }
        if not entry_levels:
            counts["standard_level_unavailable"] += 1
            level_unavailable.append(
                {
                    "category_id": category_id,
                    "name": entries[0].name,
                    "sample_level": sample_level,
                    "raw_levels": sorted({e.raw_level for e in entries}),
                    "field": _audit_field(record, field_for_audit),
                }
            )
            continue
        if sample_level in entry_levels:
            counts["matched"] += 1
        else:
            counts["mismatched"] += 1
            mismatched.append(
                {
                    "category_id": category_id,
                    "name": entries[0].name,
                    "sample_level": sample_level,
                    "standard_levels": sorted(entry_levels),
                    "standard_entry_ids": [e.standard_entry_id for e in entries],
                    "source_rows": [e.source.row for e in entries],
                    "field": _audit_field(record, field_for_audit),
                    "sample_path": list(target.get("category_path") or ()),
                }
            )

    # standard categories never observed as a resolved sample
    sample_missing = sorted(set(entries_by_category) - resolved_categories)

    # near-alias candidates for standard-missing samples (leaf-name overlap),
    # so the audit can distinguish "different standard branch" from "lost"
    for item in standard_missing:
        item["near_by_name"] = sorted(
            {
                entry.category_id
                for entry in standard.entries
                if entry.name == item["name"] and entry.category_id != item["category_id"]
            }
        )

    resolved = counts["resolved"]
    return {
        "standard": standard.dataset,
        "standard_entries": len(standard.entries),
        "training_categories": len(entries_by_category),
        "training_categories_observed": len(resolved_categories),
        "training_categories_unobserved": len(sample_missing),
        "sample_counts": {
            "total": counts["total"],
            "resolved": counts["resolved"],
            "unresolved": counts["unresolved"],
            "matched": counts["matched"],
            "mismatched": counts["mismatched"],
            "standard_missing": counts["standard_missing"],
            "standard_level_unavailable": counts["standard_level_unavailable"],
        },
        "resolved_match_rate": round(counts["matched"] / resolved, 4) if resolved else 0.0,
        "unresolved_by_status": dict(sorted(unresolved_by_status.items())),
        "unresolved_evidence": sorted(
            unresolved_evidence,
            key=lambda x: (x["status"], x["leaf_name"]),
        ),
        "mismatched_samples": sorted(mismatched, key=lambda x: (x["category_id"], x["field"])),
        "standard_missing_samples": sorted(standard_missing, key=lambda x: x["category_id"]),
        "standard_level_unavailable_samples": sorted(
            level_unavailable, key=lambda x: (x["category_id"], x["field"])
        ),
        "sample_missing_standard_categories": sample_missing,
    }


def _unresolved_evidence(
    record: Mapping[str, Any],
    status: str,
    standard: CanonicalStandard,
) -> dict[str, Any]:
    """Evidence-only: which standard entries share the record's leaf name.

    Aids the alias/near-alignment audit for unresolved samples. Never repairs.
    """
    classification = record.get("classification")
    leaf = ""
    if isinstance(classification, Mapping):
        leaf = str(classification.get("level_4", "") or "")
    candidates = sorted(
        entry.category_id
        for entry in standard.entries
        if leaf and entry.name == leaf
    )
    return {
        "status": status,
        "leaf_name": leaf,
        "candidate_standard_categories": candidates,
    }


def _audit_field(record: Mapping[str, Any], field_for_audit: str) -> str:
    metadata = record.get("metadata")
    if isinstance(metadata, Mapping):
        value = metadata.get(field_for_audit, "")
        if value:
            return str(value)
    return ""


__all__ = ["load_canonical_records", "align_dataset_to_standard"]
