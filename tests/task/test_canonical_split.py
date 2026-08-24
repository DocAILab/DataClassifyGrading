"""Canonical-layer splitting: resolution-aware exclusions, write-back,
determinism and CLI behavior."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from script.canonical.split import main, prepare_split, write_split


def _canonical_record(record_id: str, status: str, table: str = "T1") -> dict:
    record = {
        "id": record_id,
        "resolution_status": status,
        "metadata": {"table_name": table},
        "classification": {
            "level_1": "A",
            "level_2": "B",
            "level_3": "",
            "level_4": "D1",
        },
        "split": None,
        "split_exclusion_reason": None,
    }
    if status == "resolved":
        record["target"] = {
            "leaf_level": "level_4",
            "leaf_name": "D1",
            "category_id": "demo:A.B.D1",
            "category_path": ["A", "B", "D1"],
        }
    return record


def _seed_canonical(tmp_path: Path, records: list[dict]) -> Path:
    root = tmp_path / "canonical" / "demo"
    root.mkdir(parents=True)
    (root / "all.json").write_text(
        json.dumps(records, ensure_ascii=False), encoding="utf-8"
    )
    return tmp_path / "canonical"


def _resolved_records(count: int = 40) -> list[dict]:
    # spread across 5 groups so group splits have enough partitions
    return [
        _canonical_record(f"id-{i:03d}", "resolved", table=f"T{i % 5}")
        for i in range(count)
    ]


RESOLVED = _resolved_records()
MIXED = _resolved_records(30) + [
    _canonical_record("unlabeled-1", "unlabeled"),
    _canonical_record("placeholder-1", "placeholder"),
]


def test_unresolved_records_never_enter_splits(tmp_path) -> None:
    canonical = _seed_canonical(tmp_path, MIXED)
    report, records, views = prepare_split("demo", canonical_dir=canonical)
    assert report["sizes"] == {
        "train": 24,
        "val": 3,
        "test": 3,
    }
    assert report["excluded"]["count"] == 2
    assert report["excluded"]["by_reason"] == {
        "resolution_status:placeholder": 1,
        "resolution_status:unlabeled": 1,
    }
    by_id = {row["id"]: row for row in records}
    assert by_id["unlabeled-1"]["split"] is None
    assert by_id["unlabeled-1"]["split_exclusion_reason"] == (
        "resolution_status:unlabeled"
    )
    split_ids = {row["id"] for rows in views.values() for row in rows}
    assert "unlabeled-1" not in split_ids and "placeholder-1" not in split_ids
    # every pooled record carries exactly one non-null assignment
    assigned = [row for row in records if row["split"]]
    assert len(assigned) == sum(report["sizes"].values())
    assert all(row["split_exclusion_reason"] is None for row in assigned)


def test_group_key_empty_is_excluded_with_reason(tmp_path) -> None:
    records = _resolved_records(20)
    records.append(_canonical_record("no-table", "resolved", table=""))
    canonical = _seed_canonical(tmp_path, records)
    report, _, views = prepare_split(
        "demo", canonical_dir=canonical,
        split_type="group", group_key="metadata.table_name",
    )
    assert report["excluded"]["by_reason"] == {"empty_group_key:metadata.table_name": 1}
    split_ids = {row["id"] for rows in views.values() for row in rows}
    assert "no-table" not in split_ids


def test_split_write_back_is_byte_identical_on_rerun(tmp_path) -> None:
    outputs = []
    for index in range(2):
        canonical = _seed_canonical(tmp_path / f"run{index}", MIXED)
        report, records, views = prepare_split("demo", canonical_dir=canonical)
        write_split(
            "demo", canonical_dir=canonical,
            report=report, records=records, views=views, overwrite=True,
        )
        outputs.append((canonical / "demo" / "all.json").read_bytes())
        outputs.append((canonical / "demo" / "train.json").read_bytes())
    assert outputs[0] == outputs[2]
    assert outputs[1] == outputs[3]


def test_cli_end_to_end_and_overwrite_guard(tmp_path) -> None:
    canonical = _seed_canonical(tmp_path, MIXED)
    common = [
        "--canonical-dir", str(canonical),
        "--dataset", "demo",
    ]
    assert main([*common, "--overwrite"]) == 0
    root = canonical / "demo"
    for name in ("all.json", "train.json", "val.json", "test.json", "split_report.json"):
        assert (root / name).is_file()
    report = json.loads((root / "split_report.json").read_text(encoding="utf-8"))
    assert report["order_rule"] == "id-ascending"

    with pytest.raises(FileExistsError):
        main(common)

    with pytest.raises(SystemExit):
        main([
            "--canonical-dir", str(canonical),
            "--dataset", "demo",
            "--split-type", "group",
            "--overwrite",
        ])  # group without --group-key
