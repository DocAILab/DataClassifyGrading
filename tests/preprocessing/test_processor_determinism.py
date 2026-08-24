"""Determinism and label-semantics tests for the preprocessing layer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from script.preprocessing.processor import (
    apply_rewrite_rules,
    clean_label,
    detect_trailing_code,
    normalize_label,
    preprocess,
)

MAPPING = {
    "database_name": "database_name",
    "table_name": "table_name",
    "field_name": "field_name",
    "level_1": "level_1",
    "level_2": "level_2",
    "level_3": "level_3",
    "level_4": "level_4",
    "data_level": "data_level",
}

CSV_HEADER = ",".join(MAPPING)
CSV_ROWS = [
    "DB,T1,F1,A,B,C,D1,L1",
    'DB,T1,F2,A,B,C,"D2（A3）",L2',
    'DB,T2,F3,"经营 管理","技术管理","系统管理信息","配置信息",L3',
]


def _write_csv(tmp_path: Path, name: str = "input.csv") -> Path:
    source = tmp_path / name
    source.write_text(
        "\n".join([CSV_HEADER, *CSV_ROWS]) + "\n", encoding="utf-8"
    )
    return source


def _mapping_file(tmp_path: Path) -> Path:
    path = tmp_path / "mapping.json"
    path.write_text(json.dumps(MAPPING), encoding="utf-8")
    return path


def _run(tmp_path: Path, source: Path, output: Path, **kwargs) -> list[dict]:
    return preprocess(
        source,
        _mapping_file(tmp_path),
        output,
        dataset="demo",
        overwrite=True,
        **kwargs,
    )


def test_sample_id_independent_of_input_filename(tmp_path) -> None:
    first = _run(tmp_path, _write_csv(tmp_path, "one.csv"), tmp_path / "a.json")
    renamed = _write_csv(tmp_path, "completely-different-name.csv")
    second = _run(tmp_path, renamed, tmp_path / "b.json")
    assert [row["id"] for row in first] == [row["id"] for row in second]
    # dataset participates in the id
    third = preprocess(
        _write_csv(tmp_path, "one.csv"),
        _mapping_file(tmp_path),
        tmp_path / "c.json",
        dataset="other",
        overwrite=True,
    )
    assert [row["id"] for row in third] != [row["id"] for row in first]


def test_trailing_code_detected_and_kept_by_default(tmp_path) -> None:
    records = _run(tmp_path, _write_csv(tmp_path), tmp_path / "out.json")
    by_field = {row["metadata"]["field_name"]: row for row in records}
    assert detect_trailing_code("D2（A3）") == "A3"
    assert clean_label("D2（A3）") == ("D2（A3）", "A3")
    # default keeps the raw label text and records the code explicitly
    assert by_field["F2"]["classification"]["level_4"] == "D2（A3）"
    assert by_field["F2"]["label_notes"] == [
        {"field": "level_4", "kept_code": "A3"}
    ]
    assert "rewritten_from" not in by_field["F2"]
    # legacy opt-in strips the suffix from the stored label
    stripped = _run(
        tmp_path, _write_csv(tmp_path), tmp_path / "stripped.json",
        strip_trailing_codes=True,
    )
    stripped_by_field = {row["metadata"]["field_name"]: row for row in stripped}
    assert stripped_by_field["F2"]["classification"]["level_4"] == "D2"
    assert "label_notes" not in stripped_by_field["F2"]
    # normalize_label stays available as the explicit-strip helper
    assert normalize_label("D2（A3）") == "D2"


def test_rewrite_rules_are_auditable(tmp_path) -> None:
    rules = [{"field": "level_1", "match": "经营 管理", "replace": "经营管理"}]
    records = _run(
        tmp_path, _write_csv(tmp_path), tmp_path / "ruled.json",
        rewrite_rules=rules,
    )
    by_field = {row["metadata"]["field_name"]: row for row in records}
    target = by_field["F3"]
    assert target["classification"]["level_1"] == "经营管理"
    assert target["rewritten_from"] == {"level_1": "经营 管理"}
    untouched = by_field["F1"]
    assert "rewritten_from" not in untouched
    assert apply_rewrite_rules("x", None) == ("x", None)


def test_preprocess_is_byte_identical_on_rerun(tmp_path) -> None:
    outputs = []
    for index in range(2):
        destination = tmp_path / f"run{index}.json"
        _run(tmp_path, _write_csv(tmp_path), destination)
        outputs.append(destination.read_bytes())
    assert outputs[0] == outputs[1]


def test_dataset_name_required(tmp_path) -> None:
    with pytest.raises(ValueError, match="dataset"):
        preprocess(
            _write_csv(tmp_path),
            _mapping_file(tmp_path),
            tmp_path / "unused.json",
            dataset="  ",
        )
