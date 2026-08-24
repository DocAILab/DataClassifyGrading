import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from agent.task import LeafRegistry, TaskConfig
from method.dpo.preference_data import (
    build_preference_row,
    export_preferences,
    load_train_records,
    select_hard_candidates,
)


REGISTRY = LeafRegistry.from_mapping(
    {
        "categories": [
            {"category_id": label, "description": f"description {label}"}
            for label in "ABCDEFG"
        ]
    }
)
CONFIG = TaskConfig(("field_name",), "field_name_level4")
RECORD = {
    "id": "row-1",
    "metadata": {
        "field_name": "customer_email",
        "field_description": "must never leak",
        "table_name": "secret_table",
    },
    "classification": {"level_4": "C", "level_3": "secret_parent"},
    "label_status": "labeled",
}
SCORES = {"A": -3.0, "B": -0.2, "C": -0.1, "D": -0.4, "E": -0.3, "F": -2.0, "G": -1.0}


def test_select_hard_candidates_uses_top_four_wrong_and_shuffles_deterministically():
    first, rejected = select_hard_candidates(
        "C", SCORES, REGISTRY, source_id="row-1", seed=42
    )
    second, second_rejected = select_hard_candidates(
        "C", SCORES, REGISTRY, source_id="row-1", seed=42
    )

    assert first == second
    assert rejected == second_rejected == "B"
    assert set(first) == {"B", "C", "D", "E", "G"}
    assert len(first) == len(set(first)) == 5
    assert select_hard_candidates(
        "C", SCORES, REGISTRY, source_id="row-1", seed=137
    )[0] != first


def test_select_hard_candidates_breaks_score_ties_by_registry_order():
    scores = {label: -1.0 for label in REGISTRY.ids}
    candidates, rejected = select_hard_candidates(
        "C", scores, REGISTRY, source_id="row-2", seed=42
    )

    assert rejected == "A"
    assert set(candidates) == {"A", "B", "C", "D", "E"}


@pytest.mark.parametrize(
    "scores,match",
    [
        ({"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0}, "five"),
        ({"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0, "outside": 0.0}, "OOV"),
    ],
)
def test_select_hard_candidates_rejects_incomplete_or_oov_scores(scores, match):
    with pytest.raises(ValueError, match=match):
        select_hard_candidates("C", scores, REGISTRY, source_id="row-1")


def test_build_preference_row_is_field_name_only_and_exact_json():
    row = build_preference_row(RECORD, SCORES, REGISTRY, CONFIG, seed=42)

    assert row["source_id"] == "row-1"
    assert row["metadata"] == {"field_name": "customer_email"}
    assert row["chosen"] == [{"role": "assistant", "content": '{"answer":"C"}'}]
    assert row["rejected"] == [{"role": "assistant", "content": '{"answer":"B"}'}]
    assert row["ground_truth"] == "C"
    assert row["rejected_label"] == "B"
    assert row["golden_position"] == row["candidates"].index("C")
    serialized = json.dumps(row, ensure_ascii=False)
    assert "customer_email" in serialized
    assert "must never leak" not in serialized
    assert "secret_table" not in serialized
    assert "secret_parent" not in serialized


def test_build_preference_row_rejects_non_field_name_contract():
    with pytest.raises(ValueError, match="field_name"):
        build_preference_row(
            RECORD,
            SCORES,
            REGISTRY,
            TaskConfig(("field_name", "field_description")),
        )


def _write_inputs(root: Path) -> tuple[Path, Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    (root / "train.json").write_text(json.dumps([RECORD]), encoding="utf-8")
    (root / "val.json").write_text("not-json", encoding="utf-8")
    (root / "test.json").write_text("not-json", encoding="utf-8")
    registry = root / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "categories": [
                    {"category_id": item.category_id, "description": item.description}
                    for item in REGISTRY.categories
                ]
            }
        ),
        encoding="utf-8",
    )
    config = root / "task.json"
    config.write_text(
        json.dumps({"metadata_fields": ["field_name"], "task_name": "field_name_level4"}),
        encoding="utf-8",
    )
    scores = root / "scores.jsonl"
    scores.write_text(
        json.dumps({"source_id": "row-1", "scores": SCORES, "model_identity": "sft-sha"}) + "\n",
        encoding="utf-8",
    )
    return registry, config, scores


def test_load_train_records_never_reads_val_or_test(tmp_path):
    _write_inputs(tmp_path)
    assert load_train_records(tmp_path) == [RECORD]


def test_export_preferences_writes_train_only_audit_and_parquet(tmp_path):
    registry, config, scores = _write_inputs(tmp_path / "input")
    output = tmp_path / "output"

    report = export_preferences(
        tmp_path / "input", output, registry, config, scores, seed=42
    )

    assert report["requested_splits"] == ["train"]
    assert report["real_test_split_read"] is False
    assert report["metadata_fields"] == ["field_name"]
    assert report["rows"] == 1
    assert report["candidate_duplicate_rows"] == 0
    assert report["candidate_oov_rows"] == 0
    assert report["score_model_identities"] == ["sft-sha"]
    assert (output / "preferences.parquet").is_file()
    rows = pq.read_table(output / "preferences.parquet").to_pylist()
    assert rows[0]["source_id"] == "row-1"
    assert json.loads((output / "preference_report.json").read_text(encoding="utf-8")) == report


def test_export_preferences_rejects_missing_or_duplicate_score_rows(tmp_path):
    registry, config, scores = _write_inputs(tmp_path / "input")
    scores.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="missing score"):
        export_preferences(tmp_path / "input", tmp_path / "out", registry, config, scores)

    duplicate = json.dumps({"source_id": "row-1", "scores": SCORES}) + "\n"
    scores.write_text(duplicate + duplicate, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate score"):
        export_preferences(tmp_path / "input", tmp_path / "out", registry, config, scores)
