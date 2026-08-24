import json

import pytest

from agent.task import LeafRegistry
from method.dpo.evaluation import (
    build_evaluation_case,
    evaluate_cases,
    paired_classification_report,
    prediction_from_scores,
)
from method.dpo.script.evaluate import load_val_records


LABELS = ["A", "B", "C", "D", "E", "F"]


def _row(source_id, gold, prediction):
    candidates = [gold, *[label for label in LABELS if label != gold][:4]]
    return {
        "source_id": source_id,
        "ground_truth": gold,
        "prediction": prediction,
        "candidates": candidates,
        "format_valid": True,
        "contract_valid": prediction in candidates,
    }


def test_paired_report_computes_accuracy_macro_f1_and_paired_changes():
    sft = [
        _row("1", "A", "A"),
        _row("2", "B", "A"),
        _row("3", "C", "C"),
        _row("4", "D", "A"),
    ]
    dpo = [
        _row("1", "A", "A"),
        _row("2", "B", "B"),
        _row("3", "C", "A"),
        _row("4", "D", "D"),
    ]

    report = paired_classification_report(sft, dpo, LABELS)

    assert report["rows"] == 4
    assert report["sft"]["accuracy"] == pytest.approx(0.5)
    assert report["dpo"]["accuracy"] == pytest.approx(0.75)
    assert report["paired"] == {
        "sft_wrong_dpo_correct": 2,
        "sft_correct_dpo_wrong": 1,
        "both_correct": 1,
        "both_wrong": 0,
        "mcnemar_exact_p_value": 1.0,
    }
    assert report["dpo"]["macro_f1"] > report["sft"]["macro_f1"]
    assert report["sft"]["invalid"] == 0
    assert report["dpo"]["oov"] == 0
    assert report["per_class"]["B"]["support"] == 1


def test_paired_report_counts_invalid_and_oov_without_treating_them_as_labels():
    sft = [_row("1", "A", "A")]
    dpo = [_row("1", "A", "outside")]
    dpo[0]["format_valid"] = False
    dpo[0]["contract_valid"] = False

    report = paired_classification_report(sft, dpo, LABELS)

    assert report["dpo"]["accuracy"] == 0.0
    assert report["dpo"]["invalid"] == 1
    assert report["dpo"]["oov"] == 1


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda rows: rows[0].update({"source_id": "other"}), "source_id"),
        (lambda rows: rows[0].update({"ground_truth": "B"}), "ground_truth"),
        (lambda rows: rows[0].update({"candidates": ["B", "A", "C", "D", "E"]}), "candidates"),
    ],
)
def test_paired_report_rejects_unfair_pairs(mutation, match):
    sft = [_row("1", "A", "A")]
    dpo = [_row("1", "A", "A")]
    mutation(dpo)

    with pytest.raises(ValueError, match=match):
        paired_classification_report(sft, dpo, LABELS)


def test_paired_report_requires_unique_complete_source_ids():
    row = _row("1", "A", "A")
    with pytest.raises(ValueError, match="unique"):
        paired_classification_report([row, row], [row, row], LABELS)


def test_build_evaluation_case_is_deterministic_field_name_only():
    registry = LeafRegistry.from_mapping(
        {
            "categories": [
                {"category_id": label, "description": f"description {label}"}
                for label in LABELS
            ]
        }
    )
    record = {
        "id": "val-1",
        "metadata": {"field_name": "account_name", "table_name": "must_not_leak"},
        "classification": {"level_4": "A", "level_3": "must_not_leak"},
    }

    case = build_evaluation_case(record, registry, seed=137)

    assert case == build_evaluation_case(record, registry, seed=137)
    assert case["metadata"] == {"field_name": "account_name"}
    assert len(case["candidates"]) == len(set(case["candidates"])) == 5
    assert "A" in case["candidates"]
    assert "must_not_leak" not in str(case)


def test_prediction_from_scores_uses_candidate_order_for_stable_ties():
    case = {
        "source_id": "val-1",
        "ground_truth": "B",
        "candidates": ["C", "B", "A", "D", "E"],
        "prompt": [{"role": "user", "content": "choose"}],
    }
    scores = {label: -1.0 for label in case["candidates"]}

    prediction = prediction_from_scores(case, scores)

    assert prediction["prediction"] == "C"
    assert prediction["correct"] is False
    assert prediction["format_valid"] is True
    assert prediction["contract_valid"] is True


def test_evaluate_cases_resumes_complete_source_ids(tmp_path):
    output = tmp_path / "predictions.jsonl"
    existing = _row("1", "A", "A")
    output.write_text(json.dumps(existing) + "\n", encoding="utf-8")
    cases = [
        {**_row("1", "A", "A"), "prompt": []},
        {**_row("2", "B", "B"), "prompt": []},
    ]
    calls = []

    def score_fn(prompt, answers):
        calls.append((prompt, answers))
        return {label: float(index) for index, label in enumerate(answers)}

    report = evaluate_cases(cases, output, score_fn=score_fn)

    assert len(calls) == 1
    assert report == {"existing_rows": 1, "new_rows": 1, "total_rows": 2}
    assert len(output.read_text(encoding="utf-8").strip().splitlines()) == 2


def test_load_val_records_never_reads_test(tmp_path):
    record = {"id": "val-1"}
    (tmp_path / "val.json").write_text(json.dumps([record]), encoding="utf-8")
    (tmp_path / "test.json").write_text("not-json", encoding="utf-8")

    assert load_val_records(tmp_path) == [record]
