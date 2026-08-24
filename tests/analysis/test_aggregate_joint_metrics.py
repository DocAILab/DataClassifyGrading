"""Equal-macro aggregation of finance and shougang joint metrics."""

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


def test_overall_is_equal_dataset_macro_not_pooled_by_sample_count(tmp_path: Path) -> None:
    finance = _report(tmp_path / "finance.json", 0.2, 0.4)
    shougang = _report(tmp_path / "shougang.json", 0.8, 0.6)
    result = aggregate_reports({"finance": finance, "shougang": shougang})
    assert result["overall"] == {
        "joint_em": pytest.approx(0.5),
        "composite_macro_f1": pytest.approx(0.5),
        "aggregation": "equal macro over datasets",
    }
    assert set(result["datasets"]) == {"finance", "shougang"}
    assert all("report_sha256" in value for value in result["datasets"].values())


def test_cli_rejects_missing_dataset_and_writes_atomic_report(tmp_path: Path) -> None:
    finance = _report(tmp_path / "finance.json", 0.2, 0.4)
    shougang = _report(tmp_path / "shougang.json", 0.8, 0.6)
    output = tmp_path / "overall.json"
    assert main(
        [
            "--input", f"finance={finance}",
            "--input", f"shougang={shougang}",
            "--output", str(output),
        ]
    ) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["overall"]["joint_em"] == 0.5
    assert main(["--input", f"finance={finance}", "--output", str(tmp_path / "bad.json")]) == 2


def test_invalid_metric_range_is_rejected(tmp_path: Path) -> None:
    finance = _report(tmp_path / "finance.json", 1.1, 0.4)
    shougang = _report(tmp_path / "shougang.json", 0.8, 0.6)
    with pytest.raises(ValueError, match="within"):
        aggregate_reports({"finance": finance, "shougang": shougang})
