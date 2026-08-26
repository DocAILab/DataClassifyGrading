"""Atomic joint RL source-release CLI."""

from __future__ import annotations

import json
from pathlib import Path

import script.verl.rl.export as export_cli
from script.verl.rl.export import main

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "sft" / "fixtures"


def _args(output: Path, *extra: str) -> list[str]:
    return [
        "--canonical", str(FIXTURES / "canonical" / "all.json"),
        "--output-dir", str(output),
        "--dataset", "demo",
        "--registry", str(FIXTURES / "registry.json"),
        "--corpus", str(FIXTURES / "corpus.json"),
        "--task-config", str(FIXTURES / "task.json"),
        "--metadata-fields", "field_name", "table_name",
        "--grading-config", str(FIXTURES / "grading.json"),
        *extra,
    ]


def test_success_publishes_passed_validated_release(tmp_path: Path) -> None:
    output = tmp_path / "rl-release"
    assert main(_args(output)) == 0
    report = json.loads((output / "export_report.json").read_text(encoding="utf-8"))
    assert report["release"] == {"status": "passed", "published": True}
    assert report["validation"]["valid"] is True
    assert all((output / f"{split}.parquet").is_file() for split in ("train", "val", "test"))


def test_existing_release_is_never_overwritten(tmp_path: Path) -> None:
    output = tmp_path / "rl-release"
    output.mkdir()
    marker = output / "export_report.json"
    marker.write_text('{"release":"old"}', encoding="utf-8")
    assert main(_args(output)) == 2
    assert marker.read_text(encoding="utf-8") == '{"release":"old"}'


def test_failed_validation_leaves_no_release_and_writes_separate_audit(
    tmp_path: Path, monkeypatch
) -> None:
    output = tmp_path / "rl-release"
    audit = tmp_path / "failed.json"
    monkeypatch.setattr(
        export_cli,
        "validate_rl_dataset",
        lambda *args, **kwargs: {"valid": False, "synthetic": True},
        raising=False,
    )
    assert main(_args(output, "--failed-audit", str(audit))) == 2
    assert not output.exists()
    assert json.loads(audit.read_text(encoding="utf-8"))["status"] == "failed"
