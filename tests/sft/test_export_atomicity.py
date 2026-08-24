"""CLI-level SFT release publication is atomic and auditable."""

from __future__ import annotations

import json
from pathlib import Path

from script.verl.sft import export as export_cli


ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "cfg" / "task" / "leaf_registry.example.json"
CORPUS = ROOT / "cfg" / "task" / "corpus.example.json"
TASK = ROOT / "cfg" / "task" / "task.example.json"
FIELDS = ("title", "summary")


def _record(record_id: str, category: str, split: str) -> dict:
    name = category.rsplit(":", 1)[1].capitalize()
    return {
        "schema_version": 2,
        "id": record_id,
        "resolution_status": "resolved",
        "metadata": {"title": f"title {record_id}", "summary": f"summary {record_id}"},
        "classification": {"group": "Synthetic", "category": name},
        "target": {
            "leaf_level": "category",
            "leaf_name": name,
            "category_id": category,
            "category_path": ["Synthetic", name],
        },
        "split": split,
        "split_exclusion_reason": None,
    }


def _canonical(tmp_path: Path) -> Path:
    canonical = tmp_path / "canonical" / "all.json"
    canonical.parent.mkdir(parents=True)
    rows = []
    for index, category in enumerate(("demo:alpha", "demo:bravo", "demo:charlie")):
        rows.extend(
            [
                _record(f"train-{index}", category, "train"),
                _record(f"val-{index}", category, "val"),
                _record(f"test-{index}", category, "test"),
            ]
        )
    canonical.write_text(json.dumps(rows), encoding="utf-8")
    return canonical


def _run(canonical: Path, output: Path, *, failed_audit: Path | None = None) -> int:
    args = [
        "--canonical",
        str(canonical),
        "--output-dir",
        str(output),
        "--registry",
        str(REGISTRY),
        "--corpus",
        str(CORPUS),
        "--task-config",
        str(TASK),
        "--metadata-fields",
        *FIELDS,
    ]
    if failed_audit is not None:
        args.extend(("--failed-audit", str(failed_audit)))
    return export_cli.main(args)


def test_success_publishes_validated_release_without_staging_leftovers(tmp_path, capsys) -> None:
    output = tmp_path / "release"
    assert _run(_canonical(tmp_path), output) == 0

    report = json.loads((output / "export_report.json").read_text(encoding="utf-8"))
    assert report["release"]["status"] == "passed"
    assert report["validation"]["valid"] is True
    assert all((output / f"{split}.parquet").is_file() for split in ("train", "val", "test"))
    assert not list(tmp_path.glob(".release.staging-*"))
    assert "release" in capsys.readouterr().out


def test_nonempty_existing_release_is_never_overwritten_and_failure_is_audited(tmp_path) -> None:
    output = tmp_path / "release"
    output.mkdir()
    old_report = output / "export_report.json"
    old_report.write_text('{"release":{"status":"passed"}}', encoding="utf-8")
    old_bytes = old_report.read_bytes()
    audit = tmp_path / "audits" / "failed.json"

    assert _run(_canonical(tmp_path), output, failed_audit=audit) == 2
    assert old_report.read_bytes() == old_bytes
    assert json.loads(audit.read_text(encoding="utf-8"))["status"] == "failed"
    assert "non-empty" in json.loads(audit.read_text(encoding="utf-8"))["error"]


def test_passed_report_is_complete_before_atomic_rename(tmp_path, monkeypatch) -> None:
    output = tmp_path / "release"
    audit = tmp_path / "failed.json"

    original_replace = export_cli.os.replace

    def inspect_then_fail(source, destination):
        staged_report = Path(source) / "export_report.json"
        payload = json.loads(staged_report.read_text(encoding="utf-8"))
        assert payload["release"] == {
            "status": "passed",
            "published": True,
            "output_dir": str(output),
            "artifacts_sha256": payload["release"]["artifacts_sha256"],
        }
        raise OSError("synthetic rename failure")

    monkeypatch.setattr(export_cli.os, "replace", inspect_then_fail)
    assert _run(_canonical(tmp_path), output, failed_audit=audit) == 2
    assert not output.exists()
    assert not list(tmp_path.glob(".release.staging-*"))
    monkeypatch.setattr(export_cli.os, "replace", original_replace)


def test_validation_failure_cleans_staging_and_does_not_publish(tmp_path, monkeypatch) -> None:
    output = tmp_path / "release"
    audit = tmp_path / "failed.json"

    def fail_validation(*args, **kwargs):
        raise ValueError("synthetic validation failure")

    monkeypatch.setattr(export_cli, "validate_sft_dataset", fail_validation)
    assert _run(_canonical(tmp_path), output, failed_audit=audit) == 2
    assert not output.exists()
    assert json.loads(audit.read_text(encoding="utf-8"))["phase"] == "validate"
    assert not list(tmp_path.glob(".release.staging-*"))


def test_failed_audit_cannot_be_nested_in_old_release(tmp_path) -> None:
    output = tmp_path / "release"
    output.mkdir()
    (output / "old-release.marker").write_text("passed", encoding="utf-8")
    nested_audit = output / "failed.json"
    assert _run(_canonical(tmp_path), output, failed_audit=nested_audit) == 2
    assert not nested_audit.exists()
    assert (tmp_path / "release.failed.json").is_file()
