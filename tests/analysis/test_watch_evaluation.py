from __future__ import annotations

import json
from pathlib import Path

import pytest

from script.analysis.watch_evaluation import validate_report


def _report() -> dict:
    rows = [
        {
            "source_id": "a",
            "stage1_format_valid": True,
            "stage1_contract_valid": True,
            "recalled": True,
            "stage2_attempted": True,
            "stage2_format_valid": True,
            "stage2_contract_valid": True,
            "leaf_correct": True,
            "level_correct": False,
            "e2e_correct": False,
            "failures": [],
        },
        {
            "source_id": "b",
            "stage1_format_valid": True,
            "stage1_contract_valid": True,
            "recalled": False,
            "stage2_attempted": True,
            "stage2_format_valid": True,
            "stage2_contract_valid": True,
            "leaf_correct": False,
            "level_correct": True,
            "e2e_correct": False,
            "failures": [],
        },
    ]
    return {
        "metrics": {
            "sources": 2,
            "stage1_recalled_count": 1,
            "stage2_attempted": 2,
            "strict_joint_em_count": 0,
            "true_e2e_correct": 0,
        },
        "per_source": rows,
    }


def test_validate_report_recomputes_completion_counts(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    report.write_text(json.dumps(_report()), encoding="utf-8")

    result = validate_report(report, expected_sources=2)

    assert result["status"] == "completed"
    assert result["observed_sources"] == 2
    assert result["unique_source_ids"] == 2
    assert result["recomputed"] == {
        "stage1_format_valid_count": 2,
        "stage1_contract_valid_count": 2,
        "stage1_recalled_count": 1,
        "stage2_attempted_count": 2,
        "stage2_format_valid_count": 2,
        "stage2_contract_valid_count": 2,
        "leaf_correct_count": 1,
        "level_correct_count": 1,
        "joint_correct_count": 0,
        "failure_count": 0,
    }
    assert len(result["report_sha256"]) == 64


def test_validate_report_rejects_duplicate_source_ids(tmp_path: Path) -> None:
    payload = _report()
    payload["per_source"][1]["source_id"] = "a"
    report = tmp_path / "report.json"
    report.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate source_id"):
        validate_report(report, expected_sources=2)


def test_validate_report_rejects_inconsistent_reported_counts(tmp_path: Path) -> None:
    payload = _report()
    payload["metrics"]["stage1_recalled_count"] = 2
    report = tmp_path / "report.json"
    report.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="stage1_recalled_count"):
        validate_report(report, expected_sources=2)
