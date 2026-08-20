"""sample <-> canonical standard alignment audit (Phase 1).

Reads canonical records (data/canonical/<dataset>/all.json — the same records
the training pipeline consumes, unchanged) and joins them to a
``CanonicalStandard`` by canonical category identity.

Buckets (strict identity join):
- matched                     : standard exists and standard_data_level == sample data_level
- mismatched                  : standard exists, both levels known, and they differ
- standard_level_unavailable  : standard exists but its level is unparseable (null)
- standard_missing           : sample category_id not in the standard
- sample_missing             : standard category observed in no resolved sample
- unresolved                 : canonical records without a resolved target

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
    standard_by_id = standard.by_id()
    counts: Counter[str] = Counter()
    unresolved_by_status: Counter[str] = Counter()
    mismatched: list[dict[str, Any]] = []
    standard_missing: list[dict[str, Any]] = []
    level_unavailable: list[dict[str, Any]] = []
    resolved_categories: set[str] = set()

    for record in records:
        counts["total"] += 1
        status = str(record.get("resolution_status", "") or "")
        target = record.get("target")
        if status != "resolved" or not isinstance(target, Mapping):
            unresolved_by_status[status or "(no status)"] += 1
            counts["unresolved"] += 1
            continue
        category_id = str(target.get("category_id", "") or "")
        sample_level = str(record.get("data_level", "") or "")
        counts["resolved"] += 1
        resolved_categories.add(category_id)

        category = standard_by_id.get(category_id)
        if category is None:
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
        standard_level = category.standard_data_level
        if standard_level is None:
            counts["standard_level_unavailable"] += 1
            level_unavailable.append(
                {
                    "category_id": category_id,
                    "name": category.name,
                    "sample_level": sample_level,
                    "raw_level": category.raw_level,
                    "field": _audit_field(record, field_for_audit),
                }
            )
            continue
        if sample_level == standard_level:
            counts["matched"] += 1
        else:
            counts["mismatched"] += 1
            mismatched.append(
                {
                    "category_id": category_id,
                    "name": category.name,
                    "sample_level": sample_level,
                    "standard_level": standard_level,
                    "standard_raw_level": category.raw_level,
                    "source_row": category.source.row,
                    "field": _audit_field(record, field_for_audit),
                    "sample_path": list(target.get("category_path") or ()),
                }
            )

    # standard categories never observed as a resolved sample
    sample_missing = sorted(set(standard_by_id) - resolved_categories)

    # near-alias candidates for standard-missing samples (leaf-name overlap),
    # so the audit can distinguish "different standard branch" from "lost"
    for item in standard_missing:
        item["near_by_name"] = sorted(
            category_id
            for category_id, category in standard_by_id.items()
            if category.name == item["name"] and category_id != item["category_id"]
        )

    resolved = counts["resolved"]
    sample_counts = {
        "total": counts["total"],
        "resolved": counts["resolved"],
        "unresolved": counts["unresolved"],
        "matched": counts["matched"],
        "mismatched": counts["mismatched"],
        "standard_missing": counts["standard_missing"],
        "standard_level_unavailable": counts["standard_level_unavailable"],
    }
    return {
        "standard": standard.dataset,
        "standard_categories": len(standard.categories),
        "standard_categories_observed": len(resolved_categories),
        "standard_categories_unobserved": len(sample_missing),
        "sample_counts": dict(sorted(sample_counts.items())),
        "resolved_match_rate": round(counts["matched"] / resolved, 4) if resolved else 0.0,
        "unresolved_by_status": dict(sorted(unresolved_by_status.items())),
        "mismatched_samples": sorted(mismatched, key=lambda x: (x["category_id"], x["field"])),
        "standard_missing_samples": sorted(standard_missing, key=lambda x: x["category_id"]),
        "standard_level_unavailable_samples": sorted(
            level_unavailable, key=lambda x: (x["category_id"], x["field"])
        ),
        "sample_missing_standard_categories": sample_missing,
    }


def _audit_field(record: Mapping[str, Any], field_for_audit: str) -> str:
    metadata = record.get("metadata")
    if isinstance(metadata, Mapping):
        value = metadata.get(field_for_audit, "")
        if value:
            return str(value)
    return ""


__all__ = ["load_canonical_records", "align_dataset_to_standard"]
