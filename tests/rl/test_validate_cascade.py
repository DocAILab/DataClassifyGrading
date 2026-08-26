"""Formal Stage1-only shougang release validation."""

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
    expected_passthrough_materialization,
    expected_sqrt_materialization,
    validate_cascade_release,
)
from script.verl.sft.export import main as sft_export_main
from script.verl.sft.record_checkpoint import build_provenance, verify_reference_provenance

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "sft" / "fixtures"
RL_FIXTURES = ROOT / "tests" / "rl" / "fixtures"
GRADING_MANIFEST = RL_FIXTURES / "grading_manifest.json"


def _export_args(dataset: str, output: Path) -> list[str]:
    return [
        "--canonical", str(FIXTURES / "canonical" / "all.json"),
        "--output-dir", str(output),
        "--dataset", dataset,
        "--registry", str(FIXTURES / "registry.json"),
        "--corpus", str(FIXTURES / "corpus.json"),
        "--task-config", str(FIXTURES / "task.json"),
        "--metadata-fields", "field_name", "table_name", "field_description", "table_description",
        "--grading-config", str(FIXTURES / "grading.json"),
    ]


def _sft_export_args(output: Path) -> list[str]:
    return [
        "--canonical", str(FIXTURES / "canonical" / "all.json"),
        "--output-dir", str(output),
        "--registry", str(FIXTURES / "registry.json"),
        "--corpus", str(FIXTURES / "corpus.json"),
        "--task-config", str(FIXTURES / "task.json"),
        "--metadata-fields", "field_name", "table_name", "field_description", "table_description",
        "--grading-config", str(FIXTURES / "grading.json"),
    ]


def _mixture_args(family: str, source: Path, output: Path) -> list[str]:
    return [
        "--family", family,
        "--input", f"shougang={source}",
        "--grading-manifest", str(GRADING_MANIFEST),
        "--registry", str(FIXTURES / "registry.json"),
        "--corpus", str(FIXTURES / "corpus.json"),
        "--task-config", str(FIXTURES / "task.json"),
        "--metadata-fields", "field_name", "table_name", "field_description", "table_description",
        "--grading-config", str(FIXTURES / "grading.json"),
        "--output-dir", str(output),
    ]


def test_passthrough_materialization_is_singleton_and_unit_weight() -> None:
    counts, weights = expected_passthrough_materialization({"shougang": 9})
    assert counts == {"shougang": 9}
    assert weights == {"shougang": 1.0}
    # The historical name is retained only as a strict compatibility alias.
    assert expected_sqrt_materialization({"shougang": 9}) == (counts, weights)
    try:
        expected_passthrough_materialization({"finance": 1, "shougang": 9})
    except ValueError:
        pass
    else:  # pragma: no cover - assertion gives a clearer failure than pytest.raises
        raise AssertionError("joint materialization must fail closed")


def test_formal_shougang_release_is_stage1_only_and_contract_valid(tmp_path: Path) -> None:
    source = tmp_path / "shougang"
    assert export_main(_export_args("shougang", source)) == 0
    release = tmp_path / "release"
    assert mixture_main(_mixture_args("rl-cascade", source, release)) == 0

    registry = LeafRegistry.from_path(FIXTURES / "registry.json")
    task = TaskConfig.from_path(FIXTURES / "task.json")
    corpus = {
        item.category_id: item
        for item in load_corpus_categories(FIXTURES / "corpus.json")
    }
    grading_manifest = DatasetGradingManifest.from_path(GRADING_MANIFEST)
    report = validate_cascade_release(
        release,
        registry=registry,
        task_config=task,
        corpus=corpus,
        grading_manifest=grading_manifest,
    )
    assert report["valid"], report
    assert report["datasets"] == ["shougang"]
    assert report["stage2_rows"] == 0
    assert report["duplicate_source_ids"] == 0
    assert report["rows_by_dataset"]["train"] == {"shougang": 3}

    payload = json.loads((release / "export_report.json").read_text(encoding="utf-8"))
    assert payload["format"] == "dataclassify-shougang-release-v1"
    assert payload["sampling"]["policy"] == "single-dataset passthrough"
    assert set(payload["inputs"]) == {"shougang"}
    payload["sampling"]["train_achieved_weights"]["shougang"] = 0.25
    (release / "export_report.json").write_text(json.dumps(payload), encoding="utf-8")
    forged = validate_cascade_release(
        release,
        registry=registry,
        task_config=task,
        corpus=corpus,
        grading_manifest=grading_manifest,
    )
    assert not forged["valid"]
    assert any("achieved weight" in error for error in forged["errors"])

    payload["sampling"]["train_achieved_weights"]["shougang"] = 1.0
    payload["sampling"]["train_input_source_counts"]["shougang"] = 2
    payload["sampling"]["train_source_counts"]["shougang"] = 2
    (release / "export_report.json").write_text(json.dumps(payload), encoding="utf-8")
    forged_count = validate_cascade_release(
        release,
        registry=registry,
        task_config=task,
        corpus=corpus,
        grading_manifest=grading_manifest,
    )
    assert not forged_count["valid"]
    assert any("does not match actual train rows" in error for error in forged_count["errors"])

    incomplete_registry = replace(
        registry,
        categories=(replace(registry.categories[0], description=""), *registry.categories[1:]),
    )
    rejected = validate_cascade_release(
        release,
        registry=incomplete_registry,
        task_config=task,
        corpus=corpus,
        grading_manifest=grading_manifest,
    )
    assert not rejected["valid"]
    assert any("non-empty descriptions" in error for error in rejected["errors"])


def test_legacy_joint_report_and_finance_rows_fail_closed(
    tmp_path: Path,
) -> None:
    source = tmp_path / "shougang"
    assert export_main(_export_args("shougang", source)) == 0
    release = tmp_path / "release"
    assert mixture_main(_mixture_args("rl-cascade", source, release)) == 0
    report_path = release / "export_report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["format"] = "dataclassify-finance-shougang-mixture-v1"
    payload["inputs"]["finance"] = payload["inputs"]["shougang"]
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    registry = LeafRegistry.from_path(FIXTURES / "registry.json")
    task = TaskConfig.from_path(FIXTURES / "task.json")
    corpus = {
        item.category_id: item
        for item in load_corpus_categories(FIXTURES / "corpus.json")
    }
    grading_manifest = DatasetGradingManifest.from_path(GRADING_MANIFEST)
    report = validate_cascade_release(
        release,
        registry=registry,
        task_config=task,
        corpus=corpus,
        grading_manifest=grading_manifest,
    )
    assert not report["valid"]
    assert any("format" in error for error in report["errors"])
    assert any("exactly the shougang input" in error for error in report["errors"])


def test_synthetic_shougang_release_to_mixture_validation_and_provenance(
    tmp_path: Path,
) -> None:
    source = tmp_path / "shougang"
    assert sft_export_main(_sft_export_args(source)) == 0
    release = tmp_path / "release"
    assert mixture_main(_mixture_args("sft", source, release)) == 0
    mixture_report = json.loads(
        (release / "export_report.json").read_text(encoding="utf-8")
    )
    assert mixture_report["validation"]["valid"] is True

    grading_manifest = DatasetGradingManifest.from_path(GRADING_MANIFEST)
    records = json.loads((FIXTURES / "canonical" / "all.json").read_text(encoding="utf-8"))
    grading_sha = grading_manifest.sha256_for("shougang")
    bundle = audit_prompt_target_bundle(
        {"shougang": records},
        standards_by_dataset={
            "shougang": {
                "classification_standard_sha256": "b" * 64,
                "grading_standard_sha256": grading_sha,
            }
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
        release / "export_report.json",
        effective_config=effective_config,
        base_model=base,
        environment_report=environment,
        prompt_audit_report=audit_path,
        grading_manifest=GRADING_MANIFEST,
        git_commit="a" * 40,
        global_step=1,
    )
    assert provenance["prompt_identifiability"]["kind"] == "bundle"
    assert provenance["grading_manifest"]["datasets"]["shougang"]["sha256"] == grading_sha
    provenance_path = tmp_path / "provenance.json"
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    assert (
        verify_reference_provenance(provenance_path, checkpoint)["checkpoint_sha256"]
        == provenance["checkpoint_sha256"]
    )
