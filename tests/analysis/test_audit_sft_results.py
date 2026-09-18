"""Regression tests for the SFT split and prediction audit."""

from __future__ import annotations

from pathlib import Path

from script.analysis.audit_sft_results import (
    _write_figures,
    audit_predictions,
    audit_split_overlap,
    collapse_stage_rows,
    normalize_text,
    performance_slices,
)


def _record(
    source_id: str,
    *,
    field: str,
    table: str,
    description: str = "",
    table_description: str = "",
    label: str = "category:a",
    level: str = "L2",
) -> dict[str, object]:
    return {
        "source_id": source_id,
        "metadata": {
            "field_name": field,
            "table_name": table,
            "field_description": description,
            "table_description": table_description,
        },
        "ground_truth": label,
        "ground_truth_level": level,
    }


def test_normalize_text_removes_case_spacing_and_punctuation() -> None:
    assert normalize_text(" User_ID-01 ") == "userid01"
    assert normalize_text(None) == ""


def test_collapse_stage_rows_requires_one_consistent_pair() -> None:
    row = _record("s1", field="USER_ID", table="T_USER")
    stage1 = {**row, "stage": "stage1"}
    stage2 = {**row, "stage": "stage2"}
    assert collapse_stage_rows([stage1, stage2]) == [row]


def test_overlap_audit_distinguishes_ids_metadata_tables_and_near_duplicates() -> None:
    train = [
        _record(
            "train-1",
            field="USER_ID",
            table="T_USER",
            description="unique user identifier",
        ),
        _record("train-2", field="AMOUNT", table="T_ORDER"),
    ]
    test = [
        _record(
            "test-1",
            field="user-id",
            table="t user",
            description="unique user identifiers",
        ),
        _record("train-2", field="OTHER", table="T_NEW"),
    ]

    result = audit_split_overlap(train, test)

    assert result["summary"]["source_id_overlap"] == 1
    assert result["summary"]["normalized_metadata_overlap"] == 0
    assert result["summary"]["normalized_table_overlap"] == 1
    assert result["summary"]["normalized_field_table_overlap"] == 1
    assert result["per_test"][0]["nearest_train_source_id"] == "train-1"
    assert result["per_test"][0]["nearest_train_similarity"] > 0.9


def test_prediction_audit_separates_stage1_stage2_and_level_only_errors() -> None:
    test = [
        _record("a", field="A", table="T", label="cat:a", level="L1"),
        _record("b", field="B", table="T", label="cat:b", level="L2"),
        _record("c", field="C", table="U", label="cat:c", level="L3"),
    ]
    report = {
        "per_source": [
            {
                "source_id": "a",
                "ground_truth": "cat:a",
                "ground_truth_level": "L1",
                "recalled": False,
                "final_decision": "cat:x",
                "predicted_level": "L4",
                "leaf_correct": False,
                "level_correct": False,
                "e2e_correct": False,
            },
            {
                "source_id": "b",
                "ground_truth": "cat:b",
                "ground_truth_level": "L2",
                "recalled": True,
                "final_decision": "cat:x",
                "predicted_level": "L2",
                "leaf_correct": False,
                "level_correct": True,
                "e2e_correct": False,
            },
            {
                "source_id": "c",
                "ground_truth": "cat:c",
                "ground_truth_level": "L3",
                "recalled": True,
                "final_decision": "cat:c",
                "predicted_level": "L2",
                "leaf_correct": True,
                "level_correct": False,
                "e2e_correct": False,
            },
        ]
    }

    result = audit_predictions(test, report)

    assert result["summary"]["stage1_miss"] == 1
    assert result["summary"]["stage2_selection_error"] == 1
    assert result["summary"]["level_only_error"] == 1
    assert result["summary"]["stage2_oracle_accuracy_when_recalled"] == 0.5
    assert result["summary"]["category_macro_f1"] < 1.0
    assert result["errors"][0]["error_type"] == "stage1_miss+level_error"


def test_figures_are_written_as_dependency_free_vector_heatmaps(tmp_path: Path) -> None:
    prediction = {
        "category_labels": ["cat:a", "cat:b"],
        "category_confusion_matrix": [[3, 1], [0, 2]],
        "level_labels": ["L1", "L2"],
        "level_confusion_matrix": [[3, 0], [1, 2]],
    }

    _write_figures(tmp_path, prediction)

    assert (tmp_path / "category-confusion-errors.svg").read_text(
        encoding="utf-8"
    ).startswith("<svg")
    assert (tmp_path / "level-confusion.svg").read_text(
        encoding="utf-8"
    ).startswith("<svg")


def test_performance_slices_quantify_seen_tables_and_near_duplicates() -> None:
    overlap_rows = [
        {
            "source_id": "a",
            "normalized_table_overlap": True,
            "nearest_train_similarity": 1.0,
        },
        {
            "source_id": "b",
            "normalized_table_overlap": True,
            "nearest_train_similarity": 0.5,
        },
        {
            "source_id": "c",
            "normalized_table_overlap": False,
            "nearest_train_similarity": 0.4,
        },
    ]
    report_rows = [
        {"source_id": "a", "e2e_correct": True},
        {"source_id": "b", "e2e_correct": False},
        {"source_id": "c", "e2e_correct": False},
    ]

    result = performance_slices(overlap_rows, report_rows)

    assert result["seen_table"] == {"sources": 2, "correct": 1, "accuracy": 0.5}
    assert result["unseen_table"] == {"sources": 1, "correct": 0, "accuracy": 0.0}
    assert result["similarity_below_0_95"]["sources"] == 2
