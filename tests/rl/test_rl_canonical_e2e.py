"""End-to-end stage-4A validation on real data (skipped in CI without data/).

Exercises the full RL data path on the four production datasets:
data/<ds>/canonical/all.json -> resolved-only export -> VeRL v0.8.0 RL
parquet, asserting registry consistency, candidate-fidelity, determinism,
VeRL-compatible column shape and validate_rl_dataset acceptance.
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from agent.task import LeafRegistry, TaskConfig
from agent.task.canonical_dataset import load_corpus_categories
from agent.training.rl import (
    VERL_RL_COLUMNS,
    export_rl_dataset,
    validate_rl_dataset,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
REGISTRY_DIR = ROOT / "cfg" / "task" / "registry"
CORPUS_DIR = ROOT / "cfg" / "task" / "corpus"
DATASETS = ("finance", "infra", "pers_info", "shougang")
EXPECTED_CANONICAL_RESOLVED = {
    "finance": 531, "infra": 64, "pers_info": 176, "shougang": 18393
}
METADATA_FIELDS = ["field_name", "field_description"]


def _trainable_resolved(dataset: str) -> int:
    with (DATA / dataset / "canonical" / "all.json").open(encoding="utf-8") as handle:
        canonical = json.load(handle)
    by_id = {str(record.get("id")): record for record in canonical}
    split_ids: set[str] = set()
    for split in ("train", "val", "test"):
        with (DATA / dataset / f"{split}.json").open(encoding="utf-8") as handle:
            for record in json.load(handle):
                split_ids.add(str(record.get("id")))
    return sum(
        1
        for record_id in split_ids
        if by_id.get(record_id, {}).get("resolution_status") == "resolved"
    )


@pytest.fixture(scope="module")
def real_data_available() -> bool:
    return all((DATA / dataset / "canonical" / "all.json").is_file() for dataset in DATASETS)


@pytest.fixture(scope="module")
def exports(tmp_path_factory, real_data_available: bool):
    if not real_data_available:
        pytest.skip("real data/ not available")
    out_root = tmp_path_factory.mktemp("rl-e2e")
    results: dict[str, dict] = {}
    for dataset in DATASETS:
        registry_path = REGISTRY_DIR / f"{dataset}.registry.json"
        corpus_path = CORPUS_DIR / f"{dataset}.corpus.json"
        corpus = {
            category.category_id: category
            for category in load_corpus_categories(corpus_path)
        }
        config = TaskConfig.from_mapping({"metadata_fields": list(METADATA_FIELDS)})
        out = out_root / dataset
        report = export_rl_dataset(
            DATA / dataset / "canonical" / "all.json",
            DATA / dataset,
            out,
            dataset,
            registry_path,
            config,
            corpus=corpus,
        )
        results[dataset] = {
            "out": out,
            "report": report,
            "registry_path": registry_path,
            "corpus": corpus,
            "config": config,
        }
    return results


def _count_resolved(report: dict) -> int:
    return sum(
        details["exported_rows"] // 2
        for details in report["splits"].values()
    )


def test_resolved_counts_match_canonical_pipeline(exports: dict) -> None:
    for dataset, bundle in exports.items():
        assert _count_resolved(bundle["report"]) == _trainable_resolved(dataset), dataset


def test_canonical_resolved_counts_are_stable(exports: dict) -> None:
    for dataset, expected in EXPECTED_CANONICAL_RESOLVED.items():
        with (DATA / dataset / "canonical" / "all.json").open(encoding="utf-8") as handle:
            canonical = json.load(handle)
        resolved = sum(
            1 for record in canonical if record.get("resolution_status") == "resolved"
        )
        assert resolved == expected, dataset
        assert _count_resolved(exports[dataset]["report"]) <= resolved


def test_column_shape_is_verl_five(exports: dict) -> None:
    for dataset, bundle in exports.items():
        for split in ("train", "val", "test"):
            table = pq.read_table(bundle["out"] / f"{split}.parquet")
            assert set(table.column_names) == set(VERL_RL_COLUMNS), (dataset, split)


def test_every_ground_truth_belongs_to_registry(exports: dict) -> None:
    for dataset, bundle in exports.items():
        registry = LeafRegistry.from_path(bundle["registry_path"])
        for split in ("train", "val", "test"):
            rows = pq.read_table(bundle["out"] / f"{split}.parquet").to_pylist()
            for row in rows:
                assert row["reward_model"]["ground_truth"] in registry.ids, (dataset, split)
                assert row["data_source"] in {f"{dataset}/stage1", f"{dataset}/stage2"}
                assert row["reward_model"]["style"] == "rule"


def test_no_assistant_gold_and_no_algorithm_fields(exports: dict) -> None:
    for dataset, bundle in exports.items():
        rows = pq.read_table(bundle["out"] / "train.parquet").to_pylist()
        for row in rows:
            assert [message["role"] for message in row["prompt"]] == ["system", "user"], dataset
            assert not any(
                key in row for key in ("advantage", "old_logp", "rollout", "policy", "value")
            )


def test_no_unresolved_samples_in_exports(exports: dict) -> None:
    for dataset, bundle in exports.items():
        for split, details in bundle["report"]["splits"].items():
            with (DATA / dataset / f"{split}.json").open(encoding="utf-8") as handle:
                split_records = json.load(handle)
            with (DATA / dataset / "canonical" / "all.json").open(encoding="utf-8") as handle:
                canonical = json.load(handle)
            canonical_by_id = {str(record.get("id")): record for record in canonical}
            unresolved_in_split = sum(
                1
                for record in split_records
                if canonical_by_id.get(str(record.get("id")), {}).get("resolution_status")
                != "resolved"
            )
            assert details["skipped_not_resolved"] == unresolved_in_split, (dataset, split)


def test_stage1_candidate_universe_is_full_registry(exports: dict) -> None:
    for dataset, bundle in exports.items():
        registry = LeafRegistry.from_path(bundle["registry_path"])
        rows = pq.read_table(bundle["out"] / "train.parquet").to_pylist()
        stage1 = next(row for row in rows if row["extra_info"]["stage"] == "stage1")
        user = stage1["prompt"][1]["content"]
        assert user.count('"category_id"') == len(registry.categories), dataset


def test_stage2_corpus_lookup_by_category_id(exports: dict) -> None:
    for dataset, bundle in exports.items():
        rows = pq.read_table(bundle["out"] / "train.parquet").to_pylist()
        stage2 = next(row for row in rows if row["extra_info"]["stage"] == "stage2")
        for candidate in stage2["extra_info"]["candidates"]:
            assert candidate in bundle["corpus"], (dataset, candidate)


def test_validate_accepts_real_exports(exports: dict) -> None:
    for dataset, bundle in exports.items():
        report = validate_rl_dataset(
            bundle["out"],
            dataset,
            bundle["registry_path"],
            bundle["config"],
            corpus=bundle["corpus"],
        )
        assert report["valid"] is True, (dataset, report["splits"])


def test_export_is_deterministic(exports: dict, tmp_path: Path) -> None:
    for dataset, bundle in exports.items():
        out2 = tmp_path / f"{dataset}-again"
        report2 = export_rl_dataset(
            DATA / dataset / "canonical" / "all.json",
            DATA / dataset,
            out2,
            dataset,
            bundle["registry_path"],
            bundle["config"],
            corpus=bundle["corpus"],
        )
        for split in ("train", "val", "test"):
            first = (bundle["out"] / f"{split}.parquet").read_bytes()
            second = (out2 / f"{split}.parquet").read_bytes()
            assert first == second, (dataset, split)
        assert _count_resolved(report2) == _trainable_resolved(dataset), dataset
