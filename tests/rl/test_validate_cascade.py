"""Formal Stage1-only finance+shougang cascade release validation."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

from agent.task import LeafRegistry, TaskConfig
from agent.task.assets import load_corpus_categories
from agent.task.grading_manifest import DatasetGradingManifest
from agent.training.input_audit import audit_prompt_target_bundle
from script.verl.common.build_mixture import main as mixture_main
from script.verl.rl.export import main as export_main
from script.verl.rl.validate_cascade import (
    expected_sqrt_materialization,
    validate_cascade_release,
)
from script.verl.sft.export import main as sft_export_main
from script.verl.sft.record_checkpoint import build_provenance, verify_reference_provenance

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "sft" / "fixtures"


def _export_args(dataset: str, output: Path) -> list[str]:
    return [
        "--canonical", str(FIXTURES / "canonical" / "all.json"),
        "--output-dir", str(output),
        "--dataset", dataset,
        "--registry", str(FIXTURES / "registry.json"),
        "--corpus", str(FIXTURES / "corpus.json"),
        "--task-config", str(FIXTURES / "task.json"),
        "--metadata-fields", "field_name",
        "--grading-config", str(FIXTURES / "grading.json"),
    ]


def _sft_export_args(dataset: str, output: Path) -> list[str]:
    return [
        "--canonical", str(FIXTURES / "canonical" / "all.json"),
        "--output-dir", str(output),
        "--registry", str(FIXTURES / "registry.json"),
        "--corpus", str(FIXTURES / "corpus.json"),
        "--task-config", str(FIXTURES / "task.json"),
        "--metadata-fields", "field_name",
        "--grading-config", str(FIXTURES / "grading.json"),
    ]


def test_sqrt_weights_use_ceil_materialized_counts_for_non_square_inputs() -> None:
    counts, weights = expected_sqrt_materialization(
        {"finance": 425, "shougang": 14715}
    )
    assert counts == {"finance": 2501, "shougang": 14715}
    assert weights == {"finance": 2501 / 17216, "shougang": 14715 / 17216}


def test_formal_mixture_is_stage1_only_and_contract_valid(tmp_path: Path) -> None:
    finance = tmp_path / "finance"
    shougang = tmp_path / "shougang"
    assert export_main(_export_args("finance", finance)) == 0
    assert export_main(_export_args("shougang", shougang)) == 0
    mixture = tmp_path / "mixture"
    assert mixture_main(
        [
            "--family", "rl-cascade",
            "--input", f"finance={finance}",
            "--input", f"shougang={shougang}",
            "--grading-manifest", str(FIXTURES / "grading_manifest.json"),
            "--registry", str(FIXTURES / "registry.json"),
            "--corpus", str(FIXTURES / "corpus.json"),
            "--task-config", str(FIXTURES / "task.json"),
            "--metadata-fields", "field_name",
            "--grading-config", str(FIXTURES / "grading.json"),
            "--output-dir", str(mixture),
        ]
    ) == 0
    registry = LeafRegistry.from_path(FIXTURES / "registry.json")
    task = TaskConfig.from_path(FIXTURES / "task.json")
    corpus = {
        item.category_id: item
        for item in load_corpus_categories(FIXTURES / "corpus.json")
    }
    grading_manifest = DatasetGradingManifest.from_path(
        FIXTURES / "grading_manifest.json"
    )
    report = validate_cascade_release(
        mixture,
        registry=registry,
        task_config=task,
        corpus=corpus,
        grading_manifest=grading_manifest,
    )
    assert report["valid"], report
    assert report["datasets"] == ["finance", "shougang"]
    assert report["stage2_rows"] == 0
    assert report["duplicate_source_ids"] == 0

    payload = json.loads((mixture / "export_report.json").read_text(encoding="utf-8"))
    payload["sampling"]["train_achieved_weights"]["finance"] = 0.25
    (mixture / "export_report.json").write_text(json.dumps(payload), encoding="utf-8")
    forged = validate_cascade_release(
        mixture,
        registry=registry,
        task_config=task,
        corpus=corpus,
        grading_manifest=grading_manifest,
    )
    assert not forged["valid"]
    assert any("achieved weight mismatch" in error for error in forged["errors"])

    incomplete_registry = replace(
        registry,
        categories=(replace(registry.categories[0], description=""), *registry.categories[1:]),
    )
    rejected = validate_cascade_release(
        mixture,
        registry=incomplete_registry,
        task_config=task,
        corpus=corpus,
        grading_manifest=grading_manifest,
    )
    assert not rejected["valid"]
    assert any("non-empty descriptions" in error for error in rejected["errors"])


def test_synthetic_release_to_mixture_validation_and_provenance(tmp_path: Path) -> None:
    finance = tmp_path / "finance"
    shougang = tmp_path / "shougang"
    assert sft_export_main(_sft_export_args("finance", finance)) == 0
    assert sft_export_main(_sft_export_args("shougang", shougang)) == 0
    grading_manifest = DatasetGradingManifest.from_path(FIXTURES / "grading_manifest.json")
    mixture = tmp_path / "mixture"
    assert mixture_main(
        [
            "--family", "sft",
            "--input", f"finance={finance}",
            "--input", f"shougang={shougang}",
            "--grading-manifest", str(FIXTURES / "grading_manifest.json"),
            "--registry", str(FIXTURES / "registry.json"),
            "--corpus", str(FIXTURES / "corpus.json"),
            "--task-config", str(FIXTURES / "task.json"),
            "--metadata-fields", "field_name",
            "--grading-config", str(FIXTURES / "grading.json"),
            "--output-dir", str(mixture),
        ]
    ) == 0
    mixture_report = json.loads(
        (mixture / "export_report.json").read_text(encoding="utf-8")
    )
    assert mixture_report["validation"]["valid"] is True

    records = json.loads((FIXTURES / "canonical" / "all.json").read_text(encoding="utf-8"))
    grading_sha = grading_manifest.sha256_for("finance")
    bundle = audit_prompt_target_bundle(
        {"finance": records, "shougang": records},
        standards_by_dataset={
            "finance": {"classification_standard_sha256": "a" * 64, "grading_standard_sha256": grading_sha},
            "shougang": {"classification_standard_sha256": "b" * 64, "grading_standard_sha256": grading_manifest.sha256_for("shougang")},
        },
    )
    audit_path = tmp_path / "prompt-audit-bundle.json"
    audit_path.write_text(json.dumps(bundle, sort_keys=True), encoding="utf-8")

    def model(path: Path) -> Path:
        path.mkdir()
        (path / "model.safetensors").write_bytes(b"synthetic")
        (path / "config.json").write_text("{}", encoding="utf-8")
        (path / "tokenizer.json").write_text("{}", encoding="utf-8")
        return path

    checkpoint = model(tmp_path / "checkpoint")
    base = model(tmp_path / "base")
    effective_config = tmp_path / "effective.yaml"
    effective_config.write_text("trainer: synthetic\n", encoding="utf-8")
    environment = tmp_path / "environment.json"
    environment.write_text(json.dumps({"python": "synthetic"}), encoding="utf-8")
    provenance = build_provenance(
        checkpoint,
        mixture / "export_report.json",
        effective_config=effective_config,
        base_model=base,
        environment_report=environment,
        prompt_audit_report=audit_path,
        grading_manifest=FIXTURES / "grading_manifest.json",
        git_commit="a" * 40,
        global_step=1,
    )
    assert provenance["prompt_identifiability"]["kind"] == "bundle"
    assert provenance["grading_manifest"]["datasets"]["finance"]["sha256"] == grading_sha
    provenance_path = tmp_path / "provenance.json"
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    assert verify_reference_provenance(provenance_path, checkpoint)["checkpoint_sha256"] == provenance["checkpoint_sha256"]
