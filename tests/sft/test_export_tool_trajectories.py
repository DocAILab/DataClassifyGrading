"""CLI-level tool-trajectory release publication is atomic and auditable."""

from __future__ import annotations

import json
from pathlib import Path

from script.verl.sft import export_tool_trajectories as cli


ROOT = Path(__file__).resolve().parents[2]
CANONICAL = ROOT / "tests" / "sft" / "fixtures" / "tool_trajectory_canonical.json"
REGISTRY = ROOT / "tests" / "sft" / "fixtures" / "tool_registry.json"
CORPUS = ROOT / "tests" / "sft" / "fixtures" / "tool_corpus.json"
GRADING = ROOT / "tests" / "sft" / "fixtures" / "grading.json"
FIELDS = ("field_name", "table_name", "field_description", "table_description")


def _run(tmp_path: Path, *, output: str | None = None, failed_audit: Path | None = None) -> int:
    args = [
        "--canonical",
        str(CANONICAL),
        "--output-dir",
        str(tmp_path / (output or "release")),
        "--registry",
        str(REGISTRY),
        "--corpus",
        str(CORPUS),
        "--grading-config",
        str(GRADING),
        "--metadata-fields",
        *FIELDS,
    ]
    if failed_audit is not None:
        args.extend(("--failed-audit", str(failed_audit)))
    return cli.main(args)


def test_success_publishes_validated_release_without_staging_leftovers(
    tmp_path: Path,
) -> None:
    output = tmp_path / "release"
    assert _run(tmp_path) == 0

    report = json.loads((output / "export_report.json").read_text(encoding="utf-8"))
    assert report["release"]["status"] == "passed"
    assert report["validation"]["valid"] is True
    assert report["label_gap_gate"]["status"] == "passed"
    assert all((output / f"{split}.parquet").is_file() for split in ("train", "val", "test"))
    assert set(report["release"]["artifacts_sha256"]) == {
        f"{split}.parquet" for split in ("train", "val", "test")
    }
    for split in ("train", "val", "test"):
        assert (
            report["splits"][split]["parquet_sha256"]
            == report["release"]["artifacts_sha256"][f"{split}.parquet"]
        )
    assert not list(tmp_path.glob(".*.staging-*"))
    assert not (tmp_path / "release.failed.json").exists()


def test_new_collect_requires_independent_tool_think_slots(tmp_path: Path) -> None:
    collect_dir = tmp_path / "collect"
    base = [
        "--canonical",
        str(CANONICAL),
        "--registry",
        str(REGISTRY),
        "--corpus",
        str(CORPUS),
        "--grading-config",
        str(GRADING),
        "--metadata-fields",
        *FIELDS,
    ]
    assert cli.main(
        [*base, "--mode", "collect", "--collect-dir", str(collect_dir), "--shard-size", "8"]
    ) == 0
    for shard in sorted(collect_dir.glob("*.jsonl")):
        entries = [json.loads(line) for line in shard.read_text(encoding="utf-8").splitlines()]
        for entry in entries:
            entry["think"] = f"terminal reasoning for {entry['sample_id']}"
        shard.write_text(
            "\n".join(json.dumps(entry, ensure_ascii=False) for entry in entries) + "\n",
            encoding="utf-8",
        )
    output = tmp_path / "release"
    audit = tmp_path / "missing-tool-think.json"
    assert cli.main(
        [
            *base,
            "--output-dir",
            str(output),
            "--think-source",
            f"file:{collect_dir}",
            "--failed-audit",
            str(audit),
        ]
    ) == 2
    assert "tool_think slot" in json.loads(audit.read_text(encoding="utf-8"))["error"]


def test_new_schema_empty_terminal_is_refused_and_audited(tmp_path: Path) -> None:
    collect_dir = tmp_path / "collect"
    base = [
        "--canonical",
        str(CANONICAL),
        "--registry",
        str(REGISTRY),
        "--corpus",
        str(CORPUS),
        "--grading-config",
        str(GRADING),
        "--metadata-fields",
        *FIELDS,
    ]
    assert cli.main(
        [*base, "--mode", "collect", "--collect-dir", str(collect_dir), "--shard-size", "8"]
    ) == 0
    shard = sorted(collect_dir.glob("*.jsonl"))[0]
    entries = [json.loads(line) for line in shard.read_text(encoding="utf-8").splitlines()]
    entries[0]["think"] = ""
    shard.write_text(
        "\n".join(json.dumps(entry, ensure_ascii=False) for entry in entries) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "release"
    audit = tmp_path / "empty-terminal.json"
    assert cli.main(
        [
            *base,
            "--output-dir",
            str(output),
            "--think-source",
            f"file:{collect_dir}",
            "--failed-audit",
            str(audit),
        ]
    ) == 2
    assert "terminal think" in json.loads(audit.read_text(encoding="utf-8"))["error"]
    assert not output.exists()
    assert not list(tmp_path.glob(".*.staging-*"))


def test_nonempty_output_dir_is_refused_and_audited(tmp_path: Path) -> None:
    output = tmp_path / "release"
    output.mkdir()
    (output / "keep.txt").write_text("sentinel", encoding="utf-8")
    audit = tmp_path / "audit.json"
    assert _run(tmp_path, failed_audit=audit) == 2
    assert (output / "keep.txt").read_text(encoding="utf-8") == "sentinel"
    payload = json.loads(audit.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["phase"] == "prepare"
    assert "non-empty" in payload["error"]
    assert not list(tmp_path.glob(".*.staging-*"))


def test_legacy_single_string_file_assemble_cli_roundtrip(tmp_path: Path) -> None:
    collect_dir = tmp_path / "collect"
    base = [
        "--canonical",
        str(CANONICAL),
        "--registry",
        str(REGISTRY),
        "--corpus",
        str(CORPUS),
        "--grading-config",
        str(GRADING),
        "--metadata-fields",
        *FIELDS,
    ]
    assert cli.main(
        [*base, "--mode", "collect", "--collect-dir", str(collect_dir), "--shard-size", "8"]
    ) == 0
    # Explicit legacy input: remove the new tool_think field entirely and
    # provide only one terminal think string per sample.
    for shard in sorted(collect_dir.glob("*.jsonl")):
        entries = [json.loads(line) for line in shard.read_text(encoding="utf-8").splitlines()]
        for entry in entries:
            entry["think"] = f"legacy terminal reasoning for {entry['sample_id']}"
            entry.pop("tool_think", None)
            entry.pop("assistant_turns", None)
        shard.write_text(
            "\n".join(json.dumps(entry, ensure_ascii=False) for entry in entries) + "\n",
            encoding="utf-8",
        )
    output = tmp_path / "release"
    assert cli.main(
        [*base, "--output-dir", str(output), "--think-source", f"file:{collect_dir}"]
    ) == 0
    report = json.loads((output / "export_report.json").read_text(encoding="utf-8"))
    assert report["release"]["status"] == "passed"
    assert report["validation"]["valid"] is True


def test_failed_export_does_not_publish_and_writes_audit(tmp_path: Path) -> None:
    # Every resolved record without a grading label fails the export phase.
    canonical = tmp_path / "bad.json"
    records = json.loads(CANONICAL.read_text(encoding="utf-8"))
    for record in records:
        record["data_level"] = None
    canonical.write_text(json.dumps(records), encoding="utf-8")
    output = tmp_path / "release"
    audit = tmp_path / "audit.json"
    args = [
        "--canonical",
        str(canonical),
        "--output-dir",
        str(output),
        "--registry",
        str(REGISTRY),
        "--corpus",
        str(CORPUS),
        "--grading-config",
        str(GRADING),
        "--metadata-fields",
        *FIELDS,
        "--failed-audit",
        str(audit),
    ]
    assert cli.main(args) == 2
    assert not output.exists()
    payload = json.loads(audit.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["phase"] == "export"


def test_collect_then_file_assemble_cli_roundtrip(tmp_path: Path) -> None:
    collect_dir = tmp_path / "collect"
    base = [
        "--canonical",
        str(CANONICAL),
        "--registry",
        str(REGISTRY),
        "--corpus",
        str(CORPUS),
        "--grading-config",
        str(GRADING),
        "--metadata-fields",
        *FIELDS,
    ]
    assert cli.main([*base, "--mode", "collect", "--collect-dir", str(collect_dir), "--shard-size", "8"]) == 0
    shards = sorted(collect_dir.glob("*.jsonl"))
    assert len(shards) == 3
    collect_report = json.loads((collect_dir / "collect_report.json").read_text(encoding="utf-8"))
    assert collect_report["total_samples"] == 24

    # Sub-agent fills the terminal think and each ordered tool_think slot in
    # its own shard files (simulated here).
    for shard in shards:
        entries = [
            json.loads(line)
            for line in shard.read_text(encoding="utf-8").splitlines()
        ]
        for entry in entries:
            entry["think"] = f"sub-agent terminal reasoning for {entry['sample_id']}"
            entry["tool_think"] = [
                f"sub-agent tool reasoning {turn} for {entry['sample_id']}"
                for turn in range(len(entry.get("tool_calls", [])))
            ]
        shard.write_text(
            "\n".join(json.dumps(entry, ensure_ascii=False) for entry in entries) + "\n",
            encoding="utf-8",
        )

    output = tmp_path / "release"
    assert cli.main(
        [*base, "--output-dir", str(output), "--think-source", f"file:{collect_dir}"]
    ) == 0
    report = json.loads((output / "export_report.json").read_text(encoding="utf-8"))
    assert report["release"]["status"] == "passed"
    assert report["think"]["generator"] == "file"
    assert report["validation"]["valid"] is True
