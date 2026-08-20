"""Phase 1 canonical standard — sample<->standard alignment tests (hermetic)."""

from __future__ import annotations

from agent.standards.align import align_dataset_to_standard
from agent.standards.build import resolve_standard_dataset
from agent.standards.contracts import CanonicalStandard, SourceRef, StandardCategory


def _standard(categories: list[StandardCategory]) -> CanonicalStandard:
    return CanonicalStandard(
        dataset="ds",
        id_strategy="code",
        standard_source=SourceRef(),
        categories=tuple(categories),
    )


def _cat(category_id: str, name: str, level: str) -> StandardCategory:
    return StandardCategory(
        category_id=category_id, name=name, path=(name,),
        standard_data_level=level, raw_level=level,
    )


def _record(category_id: str | None, level: str, status: str = "resolved"):
    record = {"data_level": level, "resolution_status": status}
    if category_id is not None:
        record["target"] = {
            "category_id": category_id,
            "leaf_name": category_id,
            "category_path": [category_id],
        }
    return record


def test_alignment_buckets():
    standard = _standard(
        [
            _cat("A", "A", "L1"),
            _cat("B", "B", "L3"),
            _cat("C", "C", "L1"),  # observed in no sample -> sample_missing
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
    standard = _standard([_cat("A", "A", "L3")])
    records = [_record("A", "L1")]
    report = align_dataset_to_standard(records, standard)
    assert report["sample_counts"]["mismatched"] == 1
    assert records[0]["data_level"] == "L1"  # untouched


def test_alignment_standard_level_unavailable_bucket():
    standard = CanonicalStandard(
        dataset="ds", id_strategy="path", standard_source=SourceRef(),
        categories=(
            StandardCategory(
                category_id="X", name="X", standard_data_level=None, raw_level="3 4"
            ),
        ),
    )
    report = align_dataset_to_standard([_record("X", "L3")], standard)
    assert report["sample_counts"]["standard_level_unavailable"] == 1
    assert report["sample_counts"]["mismatched"] == 0


def test_dataset_standard_routing():
    assert resolve_standard_dataset("finance") == "finance"
    assert resolve_standard_dataset("shougang") == "shougang"
    assert resolve_standard_dataset("infra") == "shougang"  # shared, not a copy
    assert resolve_standard_dataset("pers_info") is None  # no confirmed standard
