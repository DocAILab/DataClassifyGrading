"""Embedded-split consumption and the label-gap export gate (schema v2)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from agent.task import LeafRegistry, TaskConfig
from agent.task.assets import load_corpus_categories
from agent.training.sft import export_sft_dataset


ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "cfg" / "task" / "leaf_registry.example.json"
CORPUS = ROOT / "cfg" / "task" / "corpus.example.json"
TASK_FIELDS = ("title", "summary")
CATEGORIES = [
    "demo:alpha",
    "demo:bravo",
    "demo:charlie",
    "demo:delta",
    "demo:echo",
    "demo:foxtrot",
]


def _canonical_record(record_id: str, category_id: str, split: str | None) -> dict:
    leaf = category_id.split(":")[1]
    display_name = leaf.capitalize()  # matches registry display names
    record = {
        "id": record_id,
        "resolution_status": "resolved",
        "metadata": {"title": f"title {record_id}", "summary": f"summary {record_id}"},
        "classification": {"group": "G", "category": display_name},
        "target": {
            "leaf_level": "category",
            "leaf_name": display_name,
            "category_id": category_id,
            "category_path": ["Synthetic", display_name],
        },
    }
    if split is not None:
        record["split"] = split
        record["split_exclusion_reason"] = None
    return record


def _seed_canonical(tmp_path: Path, records: list[dict]) -> Path:
    canonical = tmp_path / "canonical"
    canonical.mkdir(parents=True, exist_ok=True)
    (canonical / "all.json").write_text(
        json.dumps(records, ensure_ascii=False), encoding="utf-8"
    )
    return canonical / "all.json"


def _balanced_records() -> list[dict]:
    """Three labels x three splits; every label appears in train."""
    records = []
    for index, category in enumerate(CATEGORIES[:3]):
        records.append(_canonical_record(f"tr-{index}", category, "train"))
        records.append(_canonical_record(f"va-{index}", category, "val"))
        records.append(_canonical_record(f"te-{index}", category, "test"))
    return records


def _corpus_mapping():
    return {
        category.category_id: category
        for category in load_corpus_categories(CORPUS)
    }


def test_embedded_split_export_and_sha256(tmp_path) -> None:
    canonical = _seed_canonical(tmp_path, _balanced_records())
    report = export_sft_dataset(
        canonical,
        None,
        tmp_path / "sft",
        LeafRegistry.from_path(REGISTRY),
        TaskConfig.from_mapping({"metadata_fields": TASK_FIELDS}),
        corpus=_corpus_mapping(),
    )
    assert report["split_source"] == "embedded_v2"
    assert report["label_gap_gate"]["status"] == "passed"
    for split in ("train", "val", "test"):
        details = report["splits"][split]
        assert details["exported_records"] == 6  # 3 records x 2 stages
        parquet = Path(details["output_file"])
        assert parquet.is_file()
        digest = hashlib.sha256(parquet.read_bytes()).hexdigest()
        assert details["parquet_sha256"] == digest


def test_label_gap_blocks_export_without_waiver(tmp_path) -> None:
    records = _balanced_records()
    # demo:foxtrot only in test -> gate must fail
    records.append(_canonical_record("te-gap", "demo:foxtrot", "test"))
    canonical = _seed_canonical(tmp_path, records)
    with pytest.raises(ValueError, match="label-gap gate failed"):
        export_sft_dataset(
            canonical,
            None,
            tmp_path / "sft",
            LeafRegistry.from_path(REGISTRY),
            TaskConfig.from_mapping({"metadata_fields": TASK_FIELDS}),
            corpus=_corpus_mapping(),
        )


def test_label_gap_waivers_are_recorded(tmp_path) -> None:
    records = _balanced_records()
    records.append(_canonical_record("te-gap", "demo:foxtrot", "test"))
    canonical = _seed_canonical(tmp_path, records)

    common = dict(
        registry=LeafRegistry.from_path(REGISTRY),
        task_config=TaskConfig.from_mapping({"metadata_fields": TASK_FIELDS}),
        corpus=_corpus_mapping(),
    )
    whitelisted = export_sft_dataset(
        canonical,
        None,
        tmp_path / "waived-list",
        allow_label_gaps=("demo:foxtrot",),
        **common,
    )
    assert whitelisted["label_gap_gate"]["status"] == "waived"
    assert whitelisted["label_gap_gate"]["waived"] == [
        {"label": "demo:foxtrot", "split": "test"}
    ]

    catch_all = export_sft_dataset(
        canonical,
        None,
        tmp_path / "waived-all",
        allow_any_label_gap=True,
        **common,
    )
    assert catch_all["label_gap_gate"]["status"] == "waived"


def test_resolved_record_without_split_fails_fast(tmp_path) -> None:
    records = [r for r in _balanced_records() if r["id"] != "te-0"]
    records.append(_canonical_record("orphan", "demo:alpha", None))
    canonical = _seed_canonical(tmp_path, records)
    with pytest.raises(ValueError, match="without a .* assignment|split"):
        export_sft_dataset(
            canonical,
            None,
            tmp_path / "sft",
            LeafRegistry.from_path(REGISTRY),
            TaskConfig.from_mapping({"metadata_fields": TASK_FIELDS}),
            corpus=_corpus_mapping(),
        )


def test_legacy_split_dir_join_still_supported(tmp_path) -> None:
    records = _balanced_records()
    canonical = _seed_canonical(tmp_path, records)
    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    by_split = {
        "train": [r for r in records if r["id"].startswith("tr-")],
        "val": [r for r in records if r["id"].startswith("va-")],
        "test": [r for r in records if r["id"].startswith("te-")],
    }
    for name, rows in by_split.items():
        # legacy join keys on id only; strip embedded fields to prove the path
        stripped = [
            {k: v for k, v in row.items() if k not in ("split", "split_exclusion_reason")}
            for row in rows
        ]
        (split_dir / f"{name}.json").write_text(
            json.dumps(stripped, ensure_ascii=False), encoding="utf-8"
        )
    report = export_sft_dataset(
        canonical,
        split_dir,
        tmp_path / "sft-legacy",
        LeafRegistry.from_path(REGISTRY),
        TaskConfig.from_mapping({"metadata_fields": TASK_FIELDS}),
        corpus=_corpus_mapping(),
    )
    assert report["split_source"] == "split_dir_join"
    assert report["label_gap_gate"]["status"] == "passed"


def test_export_is_deterministic_across_runs(tmp_path) -> None:
    digests = []
    for index in range(2):
        canonical = _seed_canonical(tmp_path / f"run{index}", _balanced_records())
        report = export_sft_dataset(
            canonical,
            None,
            tmp_path / f"run{index}" / "sft",
            LeafRegistry.from_path(REGISTRY),
            TaskConfig.from_mapping({"metadata_fields": TASK_FIELDS}),
            corpus=_corpus_mapping(),
        )
        digests.append(report["splits"]["train"]["parquet_sha256"])
    assert digests[0] == digests[1]
