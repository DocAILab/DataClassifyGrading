import pytest

from method.sft.script.evaluate_two_stage import summarize_predictions


def test_summary_reports_retrieval_conditional_and_end_to_end_metrics() -> None:
    rows = [
        {
            "source_id": "1",
            "golden_level_4": "A",
            "stage1_contract_valid": True,
            "stage1_recalled": True,
            "stage2_attempted": True,
            "stage2_contract_valid": True,
            "prediction": "A",
        },
        {
            "source_id": "2",
            "golden_level_4": "F",
            "stage1_contract_valid": True,
            "stage1_recalled": False,
            "stage2_attempted": True,
            "stage2_contract_valid": True,
            "prediction": "B",
        },
        {
            "source_id": "3",
            "golden_level_4": "G",
            "stage1_contract_valid": False,
            "stage1_recalled": False,
            "stage2_attempted": False,
            "stage2_contract_valid": False,
            "prediction": None,
        },
    ]

    report = summarize_predictions(rows, registry_ids=("A", "B", "F", "G"))

    assert report["examples"] == 3
    assert report["stage1_contract_valid_rate"] == 2 / 3
    assert report["stage1_recall_at_5"] == 1 / 3
    assert report["stage2_attempted"] == 2
    assert report["stage2_conditional_denominator"] == 1
    assert report["stage2_conditional_accuracy"] == 1.0
    assert report["end_to_end_accuracy"] == 1 / 3
    assert report["macro_f1"] == 1 / 3


def test_summary_rejects_duplicate_source_ids() -> None:
    row = {
        "source_id": "same",
        "golden_level_4": "A",
        "stage1_contract_valid": False,
        "stage1_recalled": False,
        "stage2_attempted": False,
        "stage2_contract_valid": False,
        "prediction": None,
    }

    with pytest.raises(ValueError, match="source_id"):
        summarize_predictions([row, dict(row)], registry_ids=("A",))
