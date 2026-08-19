"""End-to-end stage-3C validation on real data (skipped in CI without data/).

Exercises the full canonical pipeline: data/<ds>/canonical/all.json ->
resolved-only export -> stage1/stage2 parquet, asserting registry
consistency, universe completeness, determinism and VeRL-readable output.
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from agent.task import LeafRegistry, PromptChoiceRegistry, TaskConfig
from agent.task.canonical_dataset import load_corpus_categories
from agent.training.sft import export_sft_dataset, validate_sft_dataset

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
    """Resolved canonical records whose id lies inside the original split
    boundaries (computed from the real data, never hard-coded)."""
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
    """Export all four real datasets once per module."""
    if not real_data_available:
        pytest.skip("real data/ not available")
    out_root = tmp_path_factory.mktemp("sft-e2e")
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
        report = export_sft_dataset(
            DATA / dataset / "canonical" / "all.json",
            DATA / dataset,
            out,
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
        details["exported_records"] // 2
        for details in report["splits"].values()
    )


def test_resolved_counts_match_canonical_pipeline(exports: dict) -> None:
    for dataset, bundle in exports.items():
        assert _count_resolved(bundle["report"]) == _trainable_resolved(dataset), dataset


def test_canonical_resolved_counts_are_stable(exports: dict) -> None:
    """Stage-3B canonical resolved counts hold; records outside the split
    boundaries are reported (data-pipeline fact), not silently added."""
    for dataset, expected in EXPECTED_CANONICAL_RESOLVED.items():
        with (DATA / dataset / "canonical" / "all.json").open(encoding="utf-8") as handle:
            canonical = json.load(handle)
        resolved = sum(
            1 for record in canonical if record.get("resolution_status") == "resolved"
        )
        assert resolved == expected, dataset
        # canonical may contain records outside the split boundaries; those are
        # never exported (they are not part of any train/val/test split)
        assert _count_resolved(exports[dataset]["report"]) <= resolved


def test_every_training_target_belongs_to_registry(exports: dict) -> None:
    for dataset, bundle in exports.items():
        registry = LeafRegistry.from_path(bundle["registry_path"])
        names = {category.name for category in registry.categories}
        for split in ("train", "val", "test"):
            rows = pq.read_table(bundle["out"] / f"{split}.parquet").to_pylist()
            for row in rows:
                assert row["ground_truth"] in registry.ids, (dataset, split, row["source_id"])
                category = registry.get(row["ground_truth"])
                assert row["ground_truth"] == category.category_id
                # ground truth is the canonical target id, not a leaf name
                assert row["ground_truth"] not in names or row["ground_truth"].startswith(
                    ("finance:", "pers_info:")
                )


def test_no_unresolved_samples_in_exports(exports: dict) -> None:
    for dataset, bundle in exports.items():
        report = bundle["report"]
        # split records that are not resolved are skipped, never exported
        for split, details in report["splits"].items():
            with (DATA / dataset / f"{split}.json").open(encoding="utf-8") as handle:
                split_records = json.load(handle)
            canonical_by_id = {}
            with (DATA / dataset / "canonical" / "all.json").open(encoding="utf-8") as handle:
                for record in json.load(handle):
                    canonical_by_id[str(record.get("id"))] = record
            unresolved_in_split = sum(
                1
                for record in split_records
                if canonical_by_id.get(str(record.get("id")), {}).get("resolution_status")
                != "resolved"
            )
            assert details["skipped_not_resolved"] == unresolved_in_split, (dataset, split)


def _stage1_catalog(user: str) -> list:
    """Extract the JSON catalog array from a Stage 1 user message."""
    body = user.split("\n", 1)[1].split("\nField metadata:", 1)[0]
    return json.loads(body)


def test_stage1_candidate_universe_is_full_registry(exports: dict) -> None:
    for dataset, bundle in exports.items():
        registry = LeafRegistry.from_path(bundle["registry_path"])
        rows = pq.read_table(bundle["out"] / "train.parquet").to_pylist()
        stage1 = next(row for row in rows if row["stage"] == "stage1")
        user = stage1["messages"][1]["content"]
        # the stage-1 catalog renders every registry category as compact
        # [choice_id, display_name] pairs; canonical ids never appear
        assert '"category_id"' not in user, dataset
        catalog = _stage1_catalog(user)
        assert isinstance(catalog, list) and len(catalog) == len(registry.categories), dataset
        assert all(isinstance(entry, list) and len(entry) == 2 for entry in catalog), dataset
        assert [entry[0] for entry in catalog] == [
            str(index) for index in range(1, len(catalog) + 1)
        ], dataset
        display_names = [entry[1] for entry in catalog]
        assert len(set(display_names)) == len(display_names), dataset


def test_finance_duplicate_leaf_names_are_disambiguated(exports: dict) -> None:
    bundle = exports["finance"]
    registry = LeafRegistry.from_path(bundle["registry_path"])
    rows = pq.read_table(bundle["out"] / "train.parquet").to_pylist()
    stage1 = next(row for row in rows if row["stage"] == "stage1")
    catalog = _stage1_catalog(stage1["messages"][1]["content"])
    # duplicate leaf names (e.g. 基本信息) render as parent-qualified suffixes
    for category in registry.categories:
        if category.name == "基本信息":
            display = next(
                entry[1] for entry in catalog if entry[1].endswith("基本信息")
            )
            assert display != "基本信息", category.category_id
            assert " / " in display
    # every display name still resolves to exactly one canonical category
    choices = PromptChoiceRegistry.from_registry(registry)
    for entry in catalog:
        assert choices.category_id_of(entry[0]) in registry.ids


def test_shougang_and_infra_short_codes_keep_canonical_contract(exports: dict) -> None:
    """Code-strategy registries (A1-1-1 ...) must keep canonical category_ids
    in ground_truth/candidates while prompts use choice ids."""
    for dataset in ("shougang", "infra"):
        bundle = exports[dataset]
        registry = LeafRegistry.from_path(bundle["registry_path"])
        # code strategy intact: every category carries its guanji code id
        assert all(category.code for category in registry.categories), dataset
        assert all(category.category_id == category.code for category in registry.categories), dataset
        rows = pq.read_table(bundle["out"] / "train.parquet").to_pylist()
        for row in rows:
            assert row["ground_truth"] in registry.ids, (dataset, row["source_id"])
            assert all(candidate in registry.ids for candidate in row["candidates"])
        stage1 = next(row for row in rows if row["stage"] == "stage1")
        assert '"category_id"' not in stage1["messages"][1]["content"]
        choices = PromptChoiceRegistry.from_registry(registry)
        # decoded assistant answer must round-trip to canonical candidates
        decoded = choices.decode_candidates(
            json.loads(stage1["messages"][-1]["content"])["candidates"]
        )
        assert list(decoded) == stage1["candidates"]


def test_stage2_corpus_lookup_by_category_id(exports: dict) -> None:
    for dataset, bundle in exports.items():
        rows = pq.read_table(bundle["out"] / "train.parquet").to_pylist()
        stage2 = next(row for row in rows if row["stage"] == "stage2")
        user = stage2["messages"][1]["content"]
        for candidate in stage2["candidates"]:
            category = bundle["corpus"].get(candidate)
            assert category is not None, (dataset, candidate)
            # bundle renders corpus name (or registry fallback name) — lookup
            # succeeded by category_id (no KeyError during export)
        assert '"name":"' in user


def test_validate_accepts_real_exports(exports: dict) -> None:
    for dataset, bundle in exports.items():
        report = validate_sft_dataset(
            bundle["out"],
            bundle["registry_path"],
            bundle["config"],
            corpus=bundle["corpus"],
        )
        assert report["valid"] is True, (dataset, report["splits"])


def test_export_is_deterministic(exports: dict, tmp_path: Path) -> None:
    for dataset, bundle in exports.items():
        out2 = tmp_path / f"{dataset}-again"
        report2 = export_sft_dataset(
            DATA / dataset / "canonical" / "all.json",
            DATA / dataset,
            out2,
            bundle["registry_path"],
            bundle["config"],
            corpus=bundle["corpus"],
        )
        for split in ("train", "val", "test"):
            first = (bundle["out"] / f"{split}.parquet").read_bytes()
            second = (out2 / f"{split}.parquet").read_bytes()
            assert first == second, (dataset, split)
        assert _count_resolved(report2) == _trainable_resolved(dataset)
