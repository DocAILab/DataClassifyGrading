"""Deterministic finance+shougang sqrt-mixture planning."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

from agent.task import GradingConfig, LeafRegistry, TaskConfig
from agent.task.assets import load_corpus_categories
from agent.task.prompt_choices import PromptChoiceRegistry
from agent.task.prompts import build_stage1_prompt, build_stage2_prompt, stage1_answer, stage2_answer
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
        metadata = {"field_name": source_id}
        candidates = build_candidates(ground_truth, REGISTRY, source_id=source_id)
        for stage in ("stage1", "stage2"):
            if stage == "stage1":
                prompt = build_stage1_prompt(metadata, REGISTRY, TASK, choices=CHOICES)
                answer = stage1_answer(candidates, choices=CHOICES)
            else:
                prompt = build_stage2_prompt(
                    metadata, candidates, REGISTRY, TASK,
                    corpus=CORPUS, choices=CHOICES, grading=GRADING,
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
                        "metadata": {"field_name": source_id},
                        "ground_truth_level": "L2",
                        **({"candidates": ["a", "b", "c", "d", "e"]} if stage == "stage2" else {}),
                    },
                }
            )
    return rows


def test_sft_mixture_keeps_pairs_and_achieves_sqrt_weights_without_dropping_large_set() -> None:
    result = build_sqrt_mixture(
        {"finance": _sft_rows("finance", 4), "shougang": _sft_rows("shougang", 16)},
        family="sft",
        split="train",
    )
    # sqrt(4):sqrt(16) = 1:2. Keeping all 16 shougang sources requires
    # 8 finance source replicas.
    assert result.source_counts == {"finance": 8, "shougang": 16}
    assert result.achieved_weights == {"finance": 1 / 3, "shougang": 2 / 3}
    assert len(result.rows) == 48
    grouped: dict[str, set[str]] = {}
    for row in result.rows:
        grouped.setdefault(row["source_id"], set()).add(row["stage"])
    assert grouped and all(stages == {"stage1", "stage2"} for stages in grouped.values())


def test_formal_rl_mixture_contains_stage1_rows_only_with_unique_episode_ids() -> None:
    result = build_sqrt_mixture(
        {"finance": _rl_rows("finance", 4), "shougang": _rl_rows("shougang", 16)},
        family="rl-cascade",
        split="train",
    )
    assert len(result.rows) == 24
    source_ids = [row["extra_info"]["source_id"] for row in result.rows]
    assert len(set(source_ids)) == len(source_ids)
    assert all(row["extra_info"]["stage"] == "stage1" for row in result.rows)
    assert all(row["data_source"].endswith("/stage1") for row in result.rows)
    assert all("candidates" not in row["extra_info"] for row in result.rows)


def test_mixture_is_input_order_independent_and_does_not_mutate_sources() -> None:
    inputs = {"finance": _sft_rows("finance", 4), "shougang": _sft_rows("shougang", 16)}
    original = deepcopy(inputs)
    first = build_sqrt_mixture(inputs, family="sft", split="train")
    reversed_inputs = {name: list(reversed(rows)) for name, rows in inputs.items()}
    second = build_sqrt_mixture(reversed_inputs, family="sft", split="train")
    assert first.rows == second.rows
    assert inputs == original


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
                    "status": "passed", "blocking": [], "blocking_levels": [],
                    "waived": [], "waived_levels": [],
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


def _mixture_args(finance: Path, shougang: Path, output: Path) -> list[str]:
    return [
        "--family", "sft",
        "--input", f"finance={finance}",
        "--input", f"shougang={shougang}",
        "--grading-manifest", str(FIXTURES / "grading_manifest.json"),
        "--registry", str(FIXTURES / "registry.json"),
        "--corpus", str(FIXTURES / "corpus.json"),
        "--task-config", str(FIXTURES / "task.json"),
        "--metadata-fields", "field_name",
        "--grading-config", str(FIXTURES / "grading.json"),
        "--output-dir", str(output),
    ]


def test_cli_atomically_publishes_hash_anchored_mixture_release(tmp_path: Path) -> None:
    finance = tmp_path / "finance"
    shougang = tmp_path / "shougang"
    _write_release(finance, "finance", 4)
    _write_release(shougang, "shougang", 16)
    output = tmp_path / "mixture"
    assert main(_mixture_args(finance, shougang, output)) == 0
    report = json.loads((output / "export_report.json").read_text(encoding="utf-8"))
    assert report["release"] == {"status": "passed", "published": True}
    assert report["sampling"]["policy"] == "p(dataset) proportional to sqrt(source_count)"
    assert report["sampling"]["train_source_counts"] == {"finance": 8, "shougang": 16}
    assert report["official_evaluation"] == "per-dataset val/test releases only"
    assert main(_mixture_args(finance, shougang, output)) == 2


def test_mixture_rejects_tampered_input_artifact_hash(tmp_path: Path) -> None:
    finance = tmp_path / "finance"
    shougang = tmp_path / "shougang"
    _write_release(finance, "finance", 2)
    _write_release(shougang, "shougang", 2)
    (finance / "train.parquet").write_bytes(b"tampered")
    assert main(_mixture_args(finance, shougang, tmp_path / "mixture")) == 2


def test_mixture_rejects_missing_grading_asset_hash(tmp_path: Path) -> None:
    finance = tmp_path / "finance"
    shougang = tmp_path / "shougang"
    _write_release(finance, "finance", 2)
    _write_release(shougang, "shougang", 2)
    report_path = finance / "export_report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["grading"].pop("standard_sha256")
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    assert main(_mixture_args(finance, shougang, tmp_path / "mixture")) == 2


def test_mixture_rejects_gap_waiver_input_release(tmp_path: Path) -> None:
    finance = tmp_path / "finance"
    shougang = tmp_path / "shougang"
    _write_release(finance, "finance", 2)
    _write_release(shougang, "shougang", 2)
    report_path = finance / "export_report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["label_gap_gate"]["status"] = "waived"
    payload["label_gap_gate"]["waived"] = [{"label": "synthetic", "split": "test"}]
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    assert main(_mixture_args(finance, shougang, tmp_path / "mixture")) == 2


def test_val_and_test_are_concatenated_once_without_sampling() -> None:
    result = build_sqrt_mixture(
        {"finance": _sft_rows("finance", 4), "shougang": _sft_rows("shougang", 16)},
        family="sft",
        split="val",
    )
    assert result.source_counts == {"finance": 4, "shougang": 16}
    assert len(result.rows) == 40
