"""Audited stable-ID migration when original tabular import is unavailable."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.task.identity import stable_record_id
from script.preprocessing.migrate_ids import migrate_processed_ids


def _record(old_id: str, field: str, *, table: str = "T") -> dict:
    return {
        "id": old_id,
        "key": field.lower(),
        "metadata": {
            "database_name": "DB",
            "table_name": table,
            "field_name": field,
            "field_description": "unchanged",
        },
        "classification": {"level_4": "Leaf"},
        "data_level": "L2",
    }


def test_migration_changes_only_id_and_writes_hash_report(tmp_path: Path) -> None:
    source = tmp_path / "old.json"
    output = tmp_path / "new.json"
    report_path = tmp_path / "report.json"
    original = [_record("legacy-1", "A"), _record("legacy-2", "B")]
    source.write_text(json.dumps(original), encoding="utf-8")
    report = migrate_processed_ids(
        source, output, dataset="demo", report_file=report_path
    )
    migrated = json.loads(output.read_text(encoding="utf-8"))
    assert [row["id"] for row in migrated] == [
        stable_record_id("demo", row["metadata"]) for row in original
    ]
    for before, after in zip(original, migrated):
        assert {key: value for key, value in after.items() if key != "id"} == {
            key: value for key, value in before.items() if key != "id"
        }
    assert report["changed_ids"] == 2
    assert report["records"] == 2
    assert report["input_sha256"] != report["output_sha256"]
    assert json.loads(report_path.read_text(encoding="utf-8")) == report


def test_migration_rejects_identity_collisions_without_output(tmp_path: Path) -> None:
    source = tmp_path / "old.json"
    output = tmp_path / "new.json"
    report = tmp_path / "report.json"
    source.write_text(
        json.dumps([_record("one", "A"), _record("two", "A")]),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="collision"):
        migrate_processed_ids(source, output, dataset="demo", report_file=report)
    assert not output.exists()
    assert not report.exists()
