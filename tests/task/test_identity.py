"""Contract tests for stable record identity."""

from __future__ import annotations

import uuid

import pytest

from agent.task.identity import stable_record_id


def test_stable_record_id_uses_dataset_and_metadata_identity() -> None:
    metadata = {
        "database_name": "DB",
        "table_name": "Table",
        "field_name": "Field",
    }
    expected = str(uuid.uuid5(uuid.NAMESPACE_URL, "demo\x1fDB\x1fTable\x1fField"))
    assert stable_record_id("demo", metadata) == expected


def test_stable_record_id_rejects_missing_identity_metadata() -> None:
    with pytest.raises(ValueError, match="database_name"):
        stable_record_id("demo", {"table_name": "Table", "field_name": "Field"})
