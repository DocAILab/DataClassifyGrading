"""Tests for raw shougang taxonomy-label auditing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from script.analysis.audit_shougang_raw_labels import (
    audit_raw_labels,
    compare_canonical_preservation,
    load_official_splits,
)


def _row(
    source_id: str,
    *,
    path: tuple[str, str, str, str],
    level: str,
) -> dict[str, object]:
    return {
        "id": source_id,
        "domain": "demo",
        "label_status": "labeled",
        "metadata": {
            "database_name": "db",
            "database_description": "",
            "table_name": f"table_{source_id}",
            "table_description": "table",
            "field_name": f"field_{source_id}",
            "field_description": "field",
            "field_type": "",
            "value": "",
        },
        "classification": {
            f"level_{index}": value for index, value in enumerate(path, start=1)
        },
        "data_level": level,
    }


def test_audit_proves_leaf_collision_but_full_paths_are_deterministic() -> None:
    splits = {
        "train": [
            _row("a", path=("d", "x", "parent-a", "——"), level="L1"),
            _row("b", path=("d", "x", "parent-b", "——"), level="L2"),
            _row("c", path=("d", "x", "parent-c", "normal"), level="L3"),
        ],
        "val": [
            _row("d", path=("d", "x", "parent-a", "——"), level="L1")
        ],
        "test": [
            _row("e", path=("d", "x", "parent-b", "——"), level="L2")
        ],
    }

    result = audit_raw_labels(splits)

    assert result["leaf_label_count"] == 2
    assert result["full_path_count"] == 3
    assert result["ambiguous_full_path_count"] == 0
    assert result["ambiguous_leaf_labels"] == {"——": ["L1", "L2"]}
    assert result["placeholder"]["records"] == 4
    assert result["placeholder"]["full_paths"] == 2
    assert result["placeholder"]["split_counts"] == {
        "test": 1,
        "train": 2,
        "val": 1,
    }
    assert all(
        len(path["level_counts"]) == 1
        for path in result["placeholder"]["path_breakdown"]
    )


def test_loader_requires_all_json_to_equal_official_split_concatenation(
    tmp_path: Path,
) -> None:
    splits = {
        "train": [_row("a", path=("d", "x", "p", "leaf"), level="L1")],
        "val": [_row("b", path=("d", "x", "p", "leaf"), level="L1")],
        "test": [_row("c", path=("d", "x", "p", "leaf"), level="L1")],
    }
    for name, rows in splits.items():
        (tmp_path / f"{name}.json").write_text(
            json.dumps(rows, ensure_ascii=False), encoding="utf-8"
        )
    (tmp_path / "all.json").write_text(
        json.dumps([*splits["train"], *splits["val"], *splits["test"]]),
        encoding="utf-8",
    )
    loaded, manifest = load_official_splits(tmp_path)
    assert {name: len(rows) for name, rows in loaded.items()} == {
        "train": 1,
        "val": 1,
        "test": 1,
    }
    assert set(manifest) == {"all.json", "test.json", "train.json", "val.json"}

    (tmp_path / "all.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="concatenation"):
        load_official_splits(tmp_path)


def test_canonical_comparison_uses_source_id_and_checks_original_fields() -> None:
    raw = _row("source-a", path=("d", "x", "p", "leaf"), level="L1")
    canonical = {
        **raw,
        "id": "canonical-a",
        "source_id": "source-a",
        "split": "train",
        "target": {"category_id": "demo:leaf"},
    }
    result = compare_canonical_preservation(
        {"train": [raw], "val": [], "test": []}, [canonical]
    )
    assert result == {
        "raw_records": 1,
        "canonical_records": 1,
        "missing_source_ids": [],
        "extra_source_ids": [],
        "split_mismatches": 0,
        "field_mismatches": {},
        "preserved": True,
    }
