"""Strict reference-checkpoint release and verification tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from agent.training.input_audit import audit_prompt_target_bundle
from script.verl.sft.record_checkpoint import (
    build_provenance,
    main,
    tree_hash,
    verify_reference_provenance,
)


def _make_hf_model(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / "model.safetensors").write_bytes(b"\x00\x01\x02")
    (root / "config.json").write_text('{"model_type":"demo"}', encoding="utf-8")
    (root / "tokenizer.json").write_text('{"version":"1.0"}', encoding="utf-8")
    return root


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_export_release(
    root: Path, *, status: str = "passed", release_status: str = "passed"
) -> Path:
    root.mkdir(parents=True)
    standard = root / "shougang-grading.json"
    standard.write_text(
        json.dumps(
            {
                "levels": ["L1", "L2", "L3", "L4"],
                "descriptions": [f"S-{index}" for index in range(1, 5)],
                "gt_field": "data_level",
            }
        ),
        encoding="utf-8",
    )
    standards = {
        "shougang": {"path": standard.name, "sha256": _sha256(standard)}
    }
    manifest = root / "grading_manifest.json"
    manifest.write_text(json.dumps({"datasets": standards}), encoding="utf-8")
    splits: dict[str, dict[str, str]] = {}
    for split in ("train", "val", "test"):
        parquet = root / f"{split}.parquet"
        parquet.write_bytes(f"{split}-rows".encode())
        splits[split] = {
            "output_file": parquet.name,
            "parquet_sha256": _sha256(parquet),
        }
    report = root / "export_report.json"
    report.write_text(
        json.dumps(
            {
                "label_gap_gate": {
                    "status": status,
                    "blocking": [],
                    "blocking_levels": [],
                    "waived": [] if status == "passed" else [{"label": "demo"}],
                    "waived_levels": [],
                },
                "release": {
                    "status": release_status,
                    "published": release_status == "passed",
                },
                "format": "dataclassify-shougang-release-v1",
                "family": "sft",
                "dataset": "shougang",
                "sampling": {
                    "policy": "single-dataset passthrough",
                    "train_input_source_counts": {"shougang": 3},
                    "train_source_counts": {"shougang": 3},
                    "train_achieved_weights": {"shougang": 1.0},
                },
                "inputs": {"shougang": {}},
                "grading_manifest": {
                    "path": manifest.name,
                    "sha256": _sha256(manifest),
                    "datasets": {
                        dataset: {"sha256": details["sha256"]}
                        for dataset, details in standards.items()
                    },
                },
                "validation": {"valid": True},
                "splits": splits,
            }
        ),
        encoding="utf-8",
    )
    return report


def _inputs(tmp_path: Path) -> dict[str, object]:
    checkpoint = _make_hf_model(tmp_path / "merged-hf")
    base_model = _make_hf_model(tmp_path / "base-hf")
    export_report = _make_export_release(tmp_path / "release")
    effective_config = tmp_path / "effective-config.yaml"
    effective_config.write_text("trainer:\n  total_epochs: 4\n", encoding="utf-8")
    environment_report = tmp_path / "environment.json"
    environment_report.write_text(
        json.dumps({"python": "3.12", "verl": "0.8.0", "torch": "2.8.0"}),
        encoding="utf-8",
    )
    prompt_audit = _make_prompt_bundle(tmp_path)
    return {
        "checkpoint_dir": checkpoint,
        "export_report": export_report,
        "effective_config": effective_config,
        "base_model": base_model,
        "environment_report": environment_report,
        "prompt_audit_report": prompt_audit,
        "grading_manifest": export_report.parent / "grading_manifest.json",
        "git_commit": "a" * 40,
        "global_step": 140,
    }


def _make_prompt_bundle(root: Path, release_name: str = "release") -> Path:
    def record(record_id: str, field_name: str, leaf: str, level: str) -> dict:
        return {
            "id": record_id,
            "resolution_status": "resolved",
            "split": "train",
            "metadata": {"field_name": field_name},
            "target": {"category_id": leaf},
            "data_level": level,
        }

    manifest = json.loads(
        (root / release_name / "grading_manifest.json").read_text(encoding="utf-8")
    )
    grading_hash = manifest["datasets"]["shougang"]["sha256"]
    report = audit_prompt_target_bundle(
        {"shougang": [record("s-one", "shougang-field", "leaf:s", "L2")]},
        standards_by_dataset={
            "shougang": {
                "classification_standard_sha256": "c" * 64,
                "grading_standard_sha256": grading_hash,
            }
        },
    )
    path = root / "prompt-audit-bundle.json"
    path.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
    return path


def test_tree_hash_is_content_and_layout_sensitive(tmp_path: Path) -> None:
    first = _make_hf_model(tmp_path / "a")
    second = _make_hf_model(tmp_path / "b")
    assert tree_hash(first)[0] == tree_hash(second)[0]

    (second / "config.json").write_text('{"model_type":"changed"}', encoding="utf-8")
    assert tree_hash(first)[0] != tree_hash(second)[0]

    renamed = _make_hf_model(tmp_path / "c")
    (renamed / "tokenizer.json").rename(renamed / "renamed.json")
    assert tree_hash(_make_hf_model(tmp_path / "d"))[0] != tree_hash(renamed)[0]


def test_provenance_verifies_release_inputs_and_final_model(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    provenance = build_provenance(**inputs)

    assert provenance["algorithm"] == "sha256-tree-v2"
    assert provenance["artifact_kind"] == "merged_hf_reference_model"
    assert provenance["git_commit"] == "a" * 40
    assert provenance["global_step"] == 140
    assert provenance["training_export_report"]["gate_status"] == "passed"
    assert set(provenance["training_export_report"]["parquet_sha256"]) == {
        "train", "val", "test"
    }
    assert provenance["effective_config"]["sha256"] == _sha256(
        inputs["effective_config"]  # type: ignore[arg-type]
    )
    assert provenance["base_model"]["sha256"] == tree_hash(
        inputs["base_model"]  # type: ignore[arg-type]
    )[0]
    assert provenance["prompt_identifiability"]["status"] == "passed"

    provenance_path = tmp_path / "reference.provenance.json"
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    verified = verify_reference_provenance(
        provenance_path, inputs["checkpoint_dir"]  # type: ignore[arg-type]
    )
    assert verified["checkpoint_sha256"] == provenance["checkpoint_sha256"]


def test_shougang_provenance_requires_and_binds_singleton_prompt_bundle(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    single = tmp_path / "single-prompt-audit.json"
    bundle = json.loads(
        Path(inputs["prompt_audit_report"]).read_text(encoding="utf-8")  # type: ignore[arg-type]
    )
    single.write_text(json.dumps(bundle["datasets"]["shougang"]), encoding="utf-8")
    with pytest.raises(ValueError, match="serialized shougang prompt-audit bundle"):
        build_provenance(**{**inputs, "prompt_audit_report": single})

    provenance = build_provenance(**inputs)
    assert provenance["prompt_identifiability"]["kind"] == "bundle"
    assert provenance["prompt_identifiability"]["datasets"] == ["shougang"]
    provenance_path = tmp_path / "shougang.provenance.json"
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    verify_reference_provenance(provenance_path, inputs["checkpoint_dir"])  # type: ignore[arg-type]
    prompt_path = Path(inputs["prompt_audit_report"])  # type: ignore[arg-type]
    prompt_path.write_text(
        prompt_path.read_text(encoding="utf-8").replace('"status": "passed"', '"status": "failed"', 1),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="prompt-identifiability bundle sha256 mismatch"):
        verify_reference_provenance(provenance_path, inputs["checkpoint_dir"])  # type: ignore[arg-type]


def test_old_joint_and_finance_release_artifacts_fail_closed(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    report = Path(inputs["export_report"])  # type: ignore[arg-type]
    payload = json.loads(report.read_text(encoding="utf-8"))
    payload["format"] = "dataclassify-finance-shougang-mixture-v1"
    payload["inputs"] = {"finance": {}, "shougang": {}}
    report.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="shougang SFT release format"):
        build_provenance(**inputs)

    payload["format"] = "dataclassify-shougang-release-v1"
    payload["inputs"] = {"finance": {}}
    payload["sampling"] = {
        "policy": "single-dataset passthrough",
        "train_input_source_counts": {"finance": 3},
        "train_source_counts": {"finance": 3},
        "train_achieved_weights": {"finance": 1.0},
    }
    report.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="inputs must be exactly shougang"):
        build_provenance(**inputs)


def test_shougang_provenance_rejects_standalone_reports_without_serialized_bundle(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    bundle = json.loads(
        Path(inputs["prompt_audit_report"]).read_text(encoding="utf-8")  # type: ignore[arg-type]
    )
    standalone = tmp_path / "shougang-prompt-audit.json"
    standalone.write_text(json.dumps(bundle["datasets"]["shougang"]), encoding="utf-8")
    with pytest.raises(ValueError, match="serialized shougang prompt-audit bundle"):
        build_provenance(**{**inputs, "prompt_audit_report": standalone})


def test_failed_or_forged_export_release_is_rejected(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    failed = _make_export_release(tmp_path / "failed-release", status="failed")
    with pytest.raises(ValueError, match="gate status"):
        build_provenance(**{**inputs, "export_report": failed})

    unpublished = _make_export_release(
        tmp_path / "unpublished-release", release_status="validated"
    )
    with pytest.raises(ValueError, match="published release status"):
        build_provenance(**{**inputs, "export_report": unpublished})

    report = Path(inputs["export_report"])  # type: ignore[arg-type]
    payload = json.loads(report.read_text(encoding="utf-8"))
    payload["splits"]["train"]["parquet_sha256"] = "0" * 64
    report.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="parquet sha256 mismatch"):
        build_provenance(**inputs)


def test_shougang_reference_rejects_gap_waivers(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    waived = _make_export_release(tmp_path / "waived-release", status="waived")
    prompt_audit = _make_prompt_bundle(tmp_path, "waived-release")
    with pytest.raises(ValueError, match="passed without waiver"):
        build_provenance(
            **{
                **inputs,
                "export_report": waived,
                "prompt_audit_report": prompt_audit,
                "grading_manifest": waived.parent / "grading_manifest.json",
            }
        )


def test_provenance_rejects_invalid_metadata_and_modified_model(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    with pytest.raises(ValueError, match="git commit"):
        build_provenance(**{**inputs, "git_commit": "not-a-commit"})
    with pytest.raises(ValueError, match="global step"):
        build_provenance(**{**inputs, "global_step": 0})
    failed_audit = tmp_path / "failed-prompt-audit.json"
    failed_audit.write_text(
        json.dumps(
            {
                "format": "field-prompt-identifiability-v1",
                "status": "failed",
                "conflict_keys": 1,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="prompt-identifiability"):
        build_provenance(**{**inputs, "prompt_audit_report": failed_audit})

    provenance = build_provenance(**inputs)
    provenance_path = tmp_path / "reference.provenance.json"
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    model = Path(inputs["checkpoint_dir"])  # type: ignore[arg-type]
    (model / "config.json").write_text('{"model_type":"tampered"}', encoding="utf-8")
    with pytest.raises(ValueError, match="checkpoint sha256 mismatch"):
        verify_reference_provenance(provenance_path, model)


def test_cli_requires_and_writes_complete_provenance_atomically(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    output = tmp_path / "reference.provenance.json"
    common = [
        "--checkpoint-dir", str(inputs["checkpoint_dir"]),
        "--export-report", str(inputs["export_report"]),
        "--effective-config", str(inputs["effective_config"]),
        "--base-model", str(inputs["base_model"]),
        "--environment-report", str(inputs["environment_report"]),
        "--prompt-audit-report", str(inputs["prompt_audit_report"]),
        "--grading-manifest", str(inputs["grading_manifest"]),
        "--git-commit", str(inputs["git_commit"]),
        "--global-step", str(inputs["global_step"]),
        "--output", str(output),
    ]
    assert main(common) == 0
    stored = json.loads(output.read_text(encoding="utf-8"))
    assert stored["checkpoint_sha256"] == tree_hash(
        inputs["checkpoint_dir"]  # type: ignore[arg-type]
    )[0]
    assert main(common) == 2
    first = output.read_text(encoding="utf-8")
    assert main([*common, "--overwrite"]) == 0
    assert output.read_text(encoding="utf-8") == first


def test_verify_rejects_minimal_forged_provenance(tmp_path: Path) -> None:
    model = _make_hf_model(tmp_path / "model")
    digest, files, total_bytes = tree_hash(model)
    forged = tmp_path / "forged.json"
    forged.write_text(
        json.dumps(
            {
                "algorithm": "sha256-tree-v2",
                "artifact_kind": "merged_hf_reference_model",
                "checkpoint_sha256": digest,
                "files": files,
                "total_bytes": total_bytes,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="required lineage fields"):
        verify_reference_provenance(forged, model)


def test_verify_rejects_mutated_lineage_files(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    provenance = build_provenance(**inputs)
    provenance_path = tmp_path / "provenance.json"
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    mutable_files = [
        Path(inputs["effective_config"]),  # type: ignore[arg-type]
        Path(inputs["environment_report"]),  # type: ignore[arg-type]
        Path(inputs["grading_manifest"]),  # type: ignore[arg-type]
        Path(inputs["prompt_audit_report"]),  # type: ignore[arg-type]
        Path(inputs["export_report"]),  # type: ignore[arg-type]
    ]
    for mutable in mutable_files:
        original = mutable.read_bytes()
        mutable.write_bytes(original + b"tamper")
        with pytest.raises((ValueError, FileNotFoundError), match="mismatch|rejected|hash|JSON|Extra"):
            verify_reference_provenance(provenance_path, inputs["checkpoint_dir"])  # type: ignore[arg-type]
        mutable.write_bytes(original)
    base = Path(inputs["base_model"])  # type: ignore[arg-type]
    (base / "config.json").write_text("tamper", encoding="utf-8")
    with pytest.raises(ValueError, match="base model provenance mismatch"):
        verify_reference_provenance(provenance_path, inputs["checkpoint_dir"])  # type: ignore[arg-type]


def test_missing_or_non_hf_checkpoint_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(NotADirectoryError):
        tree_hash(tmp_path / "nope")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no files"):
        tree_hash(empty)

    inputs = _inputs(tmp_path)
    model = Path(inputs["checkpoint_dir"])  # type: ignore[arg-type]
    (model / "tokenizer.json").unlink()
    with pytest.raises(ValueError, match="tokenizer"):
        build_provenance(**inputs)
