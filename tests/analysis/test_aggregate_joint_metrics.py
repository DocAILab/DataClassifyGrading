"""Singleton shougang metric summary."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from script.analysis.aggregate_joint_metrics import aggregate_reports, main


def _report(path: Path, em: float, f1: float) -> Path:
    path.write_text(
        json.dumps(
            {"metrics": {"strict_joint_em": em, "composite_macro_f1": f1}}
        ),
        encoding="utf-8",
    )
    return path


def test_summary_is_single_shougang_report_without_cross_dataset_average(
    tmp_path: Path,
) -> None:
    shougang = _report(tmp_path / "shougang.json", 0.8, 0.6)
    result = aggregate_reports({"shougang": shougang})
    assert result["format"] == "dataclassify-shougang-metrics-v1"
    assert result["dataset"] == "shougang"
    assert result["overall"] == {
        "joint_em": pytest.approx(0.8),
        "composite_macro_f1": pytest.approx(0.6),
        "aggregation": "single-dataset passthrough",
    }
    assert set(result["datasets"]) == {"shougang"}
    assert "report_sha256" in result["datasets"]["shougang"]


def test_cli_rejects_missing_or_extra_dataset_and_writes_atomic_report(
    tmp_path: Path,
) -> None:
    shougang = _report(tmp_path / "shougang.json", 0.8, 0.6)
    output = tmp_path / "overall.json"
    assert main(
        [
            "--input",
            f"shougang={shougang}",
            "--output",
            str(output),
        ]
    ) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["overall"]["joint_em"] == 0.8
    assert payload["dataset"] == "shougang"
    assert (
        main(
            [
                "--input",
                f"finance={shougang}",
                "--output",
                str(tmp_path / "bad.json"),
            ]
        )
        == 2
    )
    assert (
        main(
            [
                "--input",
                f"shougang={shougang}",
                "--input",
                f"finance={shougang}",
                "--output",
                str(tmp_path / "bad-extra.json"),
            ]
        )
        == 2
    )


def test_invalid_metric_range_is_rejected(tmp_path: Path) -> None:
    shougang = _report(tmp_path / "shougang.json", 1.1, 0.4)
    with pytest.raises(ValueError, match="within"):
        aggregate_reports({"shougang": shougang})
