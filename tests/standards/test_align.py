"""Phase 1 canonical standard — sample<->standard alignment tests (hermetic)."""

from __future__ import annotations

from agent.standards.align import align_dataset_to_standard
from agent.standards.build import resolve_standard_dataset
from agent.standards.contracts import CanonicalStandard, SourceRef, StandardCategory


def _standard(entries: list[StandardCategory]) -> CanonicalStandard:
    return CanonicalStandard(
        dataset="ds",
        id_strategy="code",
        standard_source=SourceRef(),
        entries=tuple(entries),
    )


def _cat(entry_id: str, level: str, category_id: str | None = None) -> StandardCategory:
    return StandardCategory(
        standard_entry_id=entry_id,
        category_id=category_id or entry_id,
        name=entry_id,
        path=(entry_id,),
        standard_data_level=level,
        raw_level=level,
    )


def _record(category_id: str | None, level: str, status: str = "resolved", leaf: str | None = None):
    record = {"data_level": level, "resolution_status": status}
    if category_id is not None:
        record["target"] = {
            "category_id": category_id,
            "leaf_name": category_id,
            "category_path": [category_id],
        }
    if leaf is not None:
        record["classification"] = {"level_4": leaf}
    return record


def test_alignment_buckets():
    standard = _standard(
        [
            _cat("A", "L1"),
            _cat("B", "L3"),
            _cat("C", "L1"),  # observed in no sample -> sample_missing
        ]
    )
    records = [
        _record("A", "L1"),                # matched
        _record("B", "L2"),                # mismatched
        _record("D", "L3"),                # resolved but standard_missing
        _record(None, "L2", "placeholder"),  # unresolved
    ]
    report = align_dataset_to_standard(records, standard)
    assert report["sample_counts"]["total"] == 4
    assert report["sample_counts"]["resolved"] == 3
    assert report["sample_counts"]["unresolved"] == 1
    assert report["sample_counts"]["matched"] == 1
    assert report["sample_counts"]["mismatched"] == 1
    assert report["sample_counts"]["standard_missing"] == 1
    assert report["standard_missing_samples"][0]["category_id"] == "D"
    assert report["sample_missing_standard_categories"] == ["C"]


def test_alignment_never_mutates_sample_level():
    standard = _standard([_cat("A", "L3")])
    records = [_record("A", "L1")]
    report = align_dataset_to_standard(records, standard)
    assert report["sample_counts"]["mismatched"] == 1
    assert records[0]["data_level"] == "L1"  # untouched


def test_alignment_standard_level_unavailable_bucket():
    standard = CanonicalStandard(
        dataset="ds", id_strategy="path", standard_source=SourceRef(),
        entries=(
            StandardCategory(
                standard_entry_id="X", category_id="X", name="X",
                standard_data_level=None, raw_level="3 4",
            ),
        ),
    )
    report = align_dataset_to_standard([_record("X", "L3")], standard)
    assert report["sample_counts"]["standard_level_unavailable"] == 1
    assert report["sample_counts"]["mismatched"] == 0


def test_alignment_multi_entry_category_matches_any_entry_level():
    # one training category backed by two distinct standard entries (L2, L3):
    # a sample at either level matches; a sample at an absent level mismatches
    standard = _standard(
        [
            _cat("finance:a.中.基本信息", "L2", category_id="finance:a.b.基本信息"),
            _cat("finance:a.乙.基本信息", "L3", category_id="finance:a.b.基本信息"),
        ]
    )
    report = align_dataset_to_standard(
        [_record("finance:a.b.基本信息", "L3")], standard
    )
    assert report["sample_counts"]["matched"] == 1
    assert report["sample_counts"]["mismatched"] == 0


def test_alignment_unresolved_evidence_is_evidence_only():
    standard = _standard([_cat("X", "L1", category_id="X")])
    records = [_record(None, "L2", "path_mismatch", leaf="X")]
    report = align_dataset_to_standard(records, standard)
    assert report["sample_counts"]["unresolved"] == 1
    assert report["unresolved_evidence"][0]["status"] == "path_mismatch"
    assert report["unresolved_evidence"][0]["leaf_name"] == "X"
    assert report["unresolved_evidence"][0]["candidate_standard_categories"] == ["X"]


def test_dataset_standard_routing():
    assert resolve_standard_dataset("finance") == "finance"
    assert resolve_standard_dataset("shougang") == "shougang"
    assert resolve_standard_dataset("infra") == "shougang"  # shared, not a copy
    assert resolve_standard_dataset("pers_info") is None  # no confirmed standard
