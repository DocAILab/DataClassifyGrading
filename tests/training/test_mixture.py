"""Formal shougang passthrough mixture tests."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from agent.task import GradingConfig, LeafRegistry, TaskConfig
from agent.task.assets import load_corpus_categories
from agent.task.prompt_choices import PromptChoiceRegistry
from agent.task.prompts import (
    build_stage1_prompt,
    build_stage2_prompt,
    stage1_answer,
    stage2_answer,
)
from agent.training.common import build_candidates
from agent.training.mixture import build_sqrt_mixture
from script.verl.common.build_mixture import main

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "sft" / "fixtures"
REGISTRY = LeafRegistry.from_path(FIXTURES / "registry.json")
TASK = TaskConfig.from_path(FIXTURES / "task.json")
CORPUS = {
    item.category_id: item
    for item in load_corpus_categories(FIXTURES / "corpus.json")
}
GRADING = GradingConfig.from_path(FIXTURES / "grading.json")
CHOICES = PromptChoiceRegistry.from_registry(REGISTRY)


def _sft_rows(dataset: str, count: int) -> list[dict]:
    rows = []
    for index in range(count):
        source_id = f"{dataset}-{index}"
        ground_truth = REGISTRY.ids[index % len(REGISTRY.ids)]
        metadata = {"field_name": source_id, "table_name": "T",
                  "field_description": "", "table_description": ""}
        candidates = build_candidates(ground_truth, REGISTRY, source_id=source_id)
        for stage in ("stage1", "stage2"):
            if stage == "stage1":
                prompt = build_stage1_prompt(metadata, REGISTRY, TASK, choices=CHOICES)
                answer = stage1_answer(candidates, choices=CHOICES)
            else:
                prompt = build_stage2_prompt(
                    metadata,
                    candidates,
                    REGISTRY,
                    TASK,
                    corpus=CORPUS,
                    choices=CHOICES,
                    grading=GRADING,
                )
                answer = stage2_answer(ground_truth, candidates, level="L2")
            rows.append(
                {
                    "source_id": source_id,
                    "stage": stage,
                    "messages": [
                        {"role": "system", "content": prompt.system},
                        {"role": "user", "content": prompt.user},
                        {"role": "assistant", "content": answer},
                    ],
                    "ground_truth": ground_truth,
                    "ground_truth_level": "L2",
                    "metadata": metadata,
                    "candidates": candidates,
                }
            )
    return rows


def _rl_rows(dataset: str, count: int) -> list[dict]:
    rows = []
    for index in range(count):
        source_id = f"{dataset}-{index}"
        for stage in ("stage1", "stage2"):
            rows.append(
                {
                    "data_source": f"{dataset}/{stage}",
                    "prompt": [{"role": "user", "content": source_id}],
                    "ability": "joint",
                    "reward_model": {"style": "rule", "ground_truth": "leaf"},
                    "extra_info": {
                        "dataset": dataset,
                        "stage": stage,
                        "source_id": source_id,
                        "metadata": {"field_name": source_id, "table_name": "T",
                        "field_description": "", "table_description": ""},
                        "ground_truth_level": "L2",
                        **(
                            {"candidates": ["a", "b", "c", "d", "e"]}
                            if stage == "stage2"
                            else {"trajectory_format": "qwen3.5-native-tools-v2"}
                        ),
                    },
                }
            )
    return rows


def test_sft_mixture_is_singleton_passthrough_and_keeps_pairs() -> None:
    rows = _sft_rows("shougang", 16)
    result = build_sqrt_mixture(
        {"shougang": rows}, family="sft", split="train"
    )
    assert result.source_counts == {"shougang": 16}
    assert result.input_source_counts == {"shougang": 16}
    assert result.achieved_weights == {"shougang": 1.0}
    assert result.sampling_policy == "single-dataset passthrough"
    assert len(result.rows) == 32
    grouped: dict[str, set[str]] = {}
    for row in result.rows:
        grouped.setdefault(row["source_id"], set()).add(row["stage"])
    assert len(grouped) == 16
    assert all(stages == {"stage1", "stage2"} for stages in grouped.values())


def test_formal_rl_mixture_projects_stage1_without_replication() -> None:
    rows = _rl_rows("shougang", 16)
    result = build_sqrt_mixture(
        {"shougang": rows}, family="rl-cascade", split="train"
    )
    assert result.source_counts == {"shougang": 16}
    assert result.achieved_weights == {"shougang": 1.0}
    assert len(result.rows) == 16
    source_ids = [row["extra_info"]["source_id"] for row in result.rows]
    assert len(set(source_ids)) == len(source_ids)
    assert all(row["extra_info"]["stage"] == "stage1" for row in result.rows)
    assert all(row["data_source"] == "shougang/stage1" for row in result.rows)
    assert all("candidates" not in row["extra_info"] for row in result.rows)


def test_mixture_is_input_order_independent_and_does_not_mutate_sources() -> None:
    inputs = {"shougang": _sft_rows("shougang", 16)}
    original = deepcopy(inputs)
    first = build_sqrt_mixture(inputs, family="sft", split="train")
    second = build_sqrt_mixture(
        {"shougang": list(reversed(inputs["shougang"]))},
        family="sft",
        split="train",
    )
    assert first.rows == second.rows
    assert inputs == original


def test_mixture_rejects_nonformal_dataset_inputs() -> None:
    with pytest.raises(ValueError, match="formal dataset set"):
        build_sqrt_mixture(
            {"finance": _sft_rows("finance", 1)}, family="sft", split="train"
        )


def _write_release(root: Path, dataset: str, count: int) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    root.mkdir()
    splits = {}
    for split in ("train", "val", "test"):
        path = root / f"{split}.parquet"
        # Real source releases have disjoint split identities; retain that
        # contract in the synthetic release helper so the mixture gate can
        # reject accidental cross-split leakage.
        pq.write_table(pa.Table.from_pylist(_sft_rows(f"{dataset}-{split}", count)), path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        splits[split] = {"output_file": str(path), "parquet_sha256": digest}
    (root / "export_report.json").write_text(
        json.dumps(
            {
                "release": {"status": "passed", "published": True},
                "label_gap_gate": {
                    "status": "passed",
                    "blocking": [],
                    "blocking_levels": [],
                    "waived": [],
                    "waived_levels": [],
                },
                "validation": {"valid": True},
                "grading": {
                    "enabled": True,
                    "levels": ["L1", "L2", "L3", "L4"],
                    "gt_field": "data_level",
                    "standard_sha256": hashlib.sha256(
                        (FIXTURES / "grading.json").read_bytes()
                    ).hexdigest(),
                },
                "splits": splits,
            }
        ),
        encoding="utf-8",
    )


def _mixture_args(shougang: Path, output: Path) -> list[str]:
    return [
        "--family",
        "sft",
        "--input",
        f"shougang={shougang}",
        "--grading-manifest",
        str(FIXTURES / "grading_manifest.json"),
        "--registry",
        str(FIXTURES / "registry.json"),
        "--corpus",
        str(FIXTURES / "corpus.json"),
        "--task-config",
        str(FIXTURES / "task.json"),
        "--metadata-fields",
        "field_name",
        "table_name",
        "field_description",
        "table_description",
        "--grading-config",
        str(FIXTURES / "grading.json"),
        "--output-dir",
        str(output),
    ]


def test_cli_atomically_publishes_singleton_mixture_release(tmp_path: Path) -> None:
    shougang = tmp_path / "shougang"
    _write_release(shougang, "shougang", 4)
    output = tmp_path / "mixture"
    assert main(_mixture_args(shougang, output)) == 0
    report = json.loads((output / "export_report.json").read_text(encoding="utf-8"))
    assert report["format"] == "dataclassify-shougang-release-v1"
    assert report["dataset"] == "shougang"
    assert report["release"] == {"status": "passed", "published": True}
    assert report["sampling"]["policy"] == "single-dataset passthrough"
    assert report["sampling"]["train_source_counts"] == {"shougang": 4}
    assert report["sampling"]["train_input_source_counts"] == {"shougang": 4}
    assert report["sampling"]["train_achieved_weights"] == {"shougang": 1.0}
    assert report["official_evaluation"] == "shougang val/test release only"
    assert main(_mixture_args(shougang, output)) == 2


def test_cli_requires_exactly_one_shougang_input(tmp_path: Path) -> None:
    shougang = tmp_path / "shougang"
    _write_release(shougang, "shougang", 2)
    with pytest.raises(ValueError, match="formal dataset set"):
        from script.verl.common.build_mixture import _inputs

        _inputs([f"finance={shougang}"])
    with pytest.raises(ValueError, match="formal dataset set"):
        from script.verl.common.build_mixture import _inputs

        _inputs([f"shougang={shougang}", f"finance={shougang}"])


def test_mixture_rejects_tampered_input_artifact_hash(tmp_path: Path) -> None:
    shougang = tmp_path / "shougang"
    _write_release(shougang, "shougang", 2)
    (shougang / "train.parquet").write_bytes(b"tampered")
    assert main(_mixture_args(shougang, tmp_path / "mixture")) == 2


def test_mixture_rejects_missing_grading_asset_hash(tmp_path: Path) -> None:
    shougang = tmp_path / "shougang"
    _write_release(shougang, "shougang", 2)
    report_path = shougang / "export_report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["grading"].pop("standard_sha256")
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    assert main(_mixture_args(shougang, tmp_path / "mixture")) == 2


def test_mixture_rejects_gap_waiver_input_release(tmp_path: Path) -> None:
    shougang = tmp_path / "shougang"
    _write_release(shougang, "shougang", 2)
    report_path = shougang / "export_report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["label_gap_gate"]["status"] = "waived"
    payload["label_gap_gate"]["waived"] = [{"label": "synthetic", "split": "test"}]
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    assert main(_mixture_args(shougang, tmp_path / "mixture")) == 2


def test_val_and_test_are_passthrough_without_sampling() -> None:
    result = build_sqrt_mixture(
        {"shougang": _sft_rows("shougang", 16)}, family="sft", split="val"
    )
    assert result.source_counts == {"shougang": 16}
    assert result.input_source_counts == {"shougang": 16}
    assert result.achieved_weights == {"shougang": 1.0}
    assert len(result.rows) == 32
