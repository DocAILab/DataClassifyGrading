import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from agent.task import CorpusCategory, LeafRegistry, TaskConfig
from agent.training.sft import export_sft_dataset, validate_sft_dataset
from script.verl.sft.export import main as export_cli
from script.verl.sft.validate import main as validate_cli


REGISTRY = {
    "categories": [
        {"category_id": "A", "name": "alpha data"},
        {"category_id": "B", "name": "beta data"},
        {"category_id": "C", "name": "gamma data"},
        {"category_id": "D", "name": "delta data"},
        {"category_id": "E", "name": "epsilon data"},
        {"category_id": "F", "name": "zeta data"},
    ]
}
CORPUS = [
    CorpusCategory(category_id="A", name="alpha data", description="alpha desc", examples=("ex1",)),
    CorpusCategory(category_id="B", name="beta data", description="beta desc"),
    CorpusCategory(category_id="C", name="gamma data", description="gamma desc"),
    CorpusCategory(category_id="D", name="delta data", description="delta desc"),
    CorpusCategory(category_id="E", name="epsilon data", description="epsilon desc"),
    CorpusCategory(category_id="F", name="zeta data", description="zeta desc"),
]
CONFIG = {"metadata_fields": ["field_name", "field_description"]}
BASE_CLASSIFICATION = {"level_1": "", "level_2": "", "level_3": "", "level_4": "C"}


def _canonical_record(record_id: str, category_id: str = "C", status: str = "resolved"):
    return {
        "id": record_id,
        "metadata": {"field_name": "email", "field_description": "contact address", "table_name": "secret_table"},
        "classification": {**BASE_CLASSIFICATION, "level_4": "C"},
        "data_level": "L2",
        "label_status": "labeled",
        "resolution_status": status,
        "target": {
            "leaf_level": "level_4",
            "leaf_name": "gamma data",
            "category_id": category_id,
            "category_path": ["C"],
        },
    }


def _write_inputs(root: Path, canonical_records: dict[str, dict] | None = None):
    """Writes split JSON (id boundaries) + canonical/all.json + registry + task."""
    root.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        record_id = f"row-{split}"
        (root / f"{split}.json").write_text(
            json.dumps([{"id": record_id}]), encoding="utf-8"
        )
    canonical = canonical_records or {
        f"row-{split}": _canonical_record(f"row-{split}") for split in ("train", "val", "test")
    }
    canonical_dir = root / "canonical"
    canonical_dir.mkdir(parents=True, exist_ok=True)
    (canonical_dir / "all.json").write_text(
        json.dumps(list(canonical.values()), ensure_ascii=False), encoding="utf-8"
    )
    registry = root / "registry.json"
    registry.write_text(json.dumps(REGISTRY), encoding="utf-8")
    config = root / "task.json"
    config.write_text(json.dumps(CONFIG), encoding="utf-8")
    corpus = root / "corpus.json"
    corpus.write_text(
        json.dumps(
            {
                "categories": [
                    {
                        "category_id": c.category_id,
                        "name": c.name,
                        "description": c.description,
                        "path": [],
                        "code": None,
                        "examples": list(c.examples),
                    }
                    for c in CORPUS
                ]
            }
        ),
        encoding="utf-8",
    )
    return registry, config, corpus


def _corpus_map():
    return {category.category_id: category for category in CORPUS}


def test_exporter_writes_messages_parquet_and_report(tmp_path):
    registry_path, config_path, _ = _write_inputs(tmp_path / "input")
    out = tmp_path / "out"
    report = export_sft_dataset(
        tmp_path / "input" / "canonical" / "all.json",
        tmp_path / "input",
        out,
        registry_path,
        config_path,
        corpus=_corpus_map(),
    )
    assert report["format"] == "verl_sft_messages_parquet"
    assert report["label_source"].startswith("canonical target.category_id")
    assert report["splits"]["train"]["exported_records"] == 2
    assert report["splits"]["train"]["skipped_not_resolved"] == 0
    table = pq.read_table(out / "train.parquet")
    rows = table.to_pylist()
    assert len(rows) == 2
    assert {row["stage"] for row in rows} == {"stage1", "stage2"}
    for row in rows:
        assert [message["role"] for message in row["messages"]] == ["system", "user", "assistant"]
        assert "<|im_start|>" not in json.dumps(row["messages"])
        assert row["ground_truth"] == "C"
    assert (out / "export_report.json").is_file()
    assert validate_sft_dataset(
        out, registry_path, config_path, corpus=_corpus_map()
    )["valid"] is True


def test_stage1_prompt_catalog_contains_id_and_name(tmp_path):
    registry_path, config_path, _ = _write_inputs(tmp_path / "input")
    out = tmp_path / "out"
    export_sft_dataset(
        tmp_path / "input" / "canonical" / "all.json",
        tmp_path / "input",
        out,
        registry_path,
        config_path,
        corpus=_corpus_map(),
    )
    rows = pq.read_table(out / "train.parquet").to_pylist()
    stage1 = next(row for row in rows if row["stage"] == "stage1")
    user = stage1["messages"][1]["content"]
    assert '"category_id": "A"' in user and '"name": "alpha data"' in user
    assert "gamma desc" not in user  # descriptions stay out of Stage 1


def test_stage2_prompt_resolves_corpus_by_category_id(tmp_path):
    registry_path, config_path, corpus_path = _write_inputs(tmp_path / "input")
    out = tmp_path / "out"
    export_sft_dataset(
        tmp_path / "input" / "canonical" / "all.json",
        tmp_path / "input",
        out,
        registry_path,
        config_path,
        corpus=_corpus_map(),
    )
    rows = pq.read_table(out / "train.parquet").to_pylist()
    stage2 = next(row for row in rows if row["stage"] == "stage2")
    user = stage2["messages"][1]["content"]
    assert '"name":"alpha data"' in user
    assert '"descriptions":[]' in user and '"examples":["ex1"]' in user


def test_candidate_construction_is_deterministic_and_gt_first():
    from agent.training.sft import build_candidates

    registry = LeafRegistry.from_mapping(REGISTRY)
    expected = ["D", "A", "B", "C", "E"]
    assert build_candidates("D", registry) == expected
    assert build_candidates("D", registry) == expected


def test_unresolved_records_never_enter_training(tmp_path):
    canonical = {
        "row-train": _canonical_record("row-train", status="resolved"),
        "row-val": _canonical_record("row-val", status="missing_leaf"),
        "row-test": _canonical_record("row-test", status="resolved"),
    }
    # add a second val record that IS resolved so the split is not empty
    canonical["row-val-2"] = _canonical_record("row-val-2", status="resolved")
    (tmp_path / "input").mkdir(parents=True, exist_ok=True) if False else None
    registry_path, config_path, _ = _write_inputs(tmp_path / "input", canonical)
    # split boundaries: val contains both records
    (tmp_path / "input" / "val.json").write_text(
        json.dumps([{"id": "row-val"}, {"id": "row-val-2"}]), encoding="utf-8"
    )
    out = tmp_path / "out"
    report = export_sft_dataset(
        tmp_path / "input" / "canonical" / "all.json",
        tmp_path / "input",
        out,
        registry_path,
        config_path,
        corpus=_corpus_map(),
    )
    assert report["splits"]["train"]["exported_records"] == 2
    # only the resolved val record enters training; the unresolved one is skipped
    assert report["splits"]["val"]["exported_records"] == 2
    assert report["splits"]["val"]["skipped_not_resolved"] == 1
    val_rows = pq.read_table(out / "val.parquet").to_pylist()
    assert {row["source_id"] for row in val_rows} == {"row-val-2"}


def test_exporter_rejects_resolved_record_without_target(tmp_path):
    canonical = {"row-train": _canonical_record("row-train")}
    del canonical["row-train"]["target"]
    registry_path, config_path, _ = _write_inputs(tmp_path / "input", canonical)

    with pytest.raises(ValueError, match="resolved but has no target"):
        export_sft_dataset(
            tmp_path / "input" / "canonical" / "all.json",
            tmp_path / "input",
            tmp_path / "out",
            registry_path,
            config_path,
            corpus=_corpus_map(),
        )


def test_exporter_rejects_target_id_absent_from_registry(tmp_path):
    canonical = {"row-train": _canonical_record("row-train", category_id="Z")}
    registry_path, config_path, _ = _write_inputs(tmp_path / "input", canonical)

    with pytest.raises(ValueError, match="absent from the leaf registry"):
        export_sft_dataset(
            tmp_path / "input" / "canonical" / "all.json",
            tmp_path / "input",
            tmp_path / "out",
            registry_path,
            config_path,
            corpus=_corpus_map(),
        )


def test_exporter_rejects_leaf_name_registry_mismatch(tmp_path):
    record = _canonical_record("row-train")
    record["target"]["leaf_name"] = "wrong name"
    registry_path, config_path, _ = _write_inputs(tmp_path / "input", {"row-train": record})

    with pytest.raises(ValueError, match="does not match registry category"):
        export_sft_dataset(
            tmp_path / "input" / "canonical" / "all.json",
            tmp_path / "input",
            tmp_path / "out",
            registry_path,
            config_path,
            corpus=_corpus_map(),
        )


def test_exporter_rejects_split_id_absent_from_canonical(tmp_path):
    registry_path, config_path, _ = _write_inputs(tmp_path / "input")
    (tmp_path / "input" / "train.json").write_text(
        json.dumps([{"id": "row-train"}, {"id": "ghost-id"}]), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="absent from the canonical dataset"):
        export_sft_dataset(
            tmp_path / "input" / "canonical" / "all.json",
            tmp_path / "input",
            tmp_path / "out",
            registry_path,
            config_path,
            corpus=_corpus_map(),
        )


def test_idless_non_resolved_audit_records_are_allowed(tmp_path):
    """Stage-3B contract allows id-less audit records for non-resolved
    outcomes; they must never fail the export and never enter training."""
    canonical = {
        "row-train": _canonical_record("row-train", status="resolved"),
        "row-val": _canonical_record("row-val", status="resolved"),
        "row-test": _canonical_record("row-test", status="resolved"),
    }
    registry_path, config_path, _ = _write_inputs(tmp_path / "input", canonical)
    # append id-less audit records exactly as canonical_dataset emits them
    with (tmp_path / "input" / "canonical" / "all.json").open("a", encoding="utf-8") as handle:
        pass  # rewrite below instead of appending (valid JSON)
    with (tmp_path / "input" / "canonical" / "all.json").open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    records.append({"record": "not-a-record", "resolution_status": "invalid_record"})
    records.append({"resolution_status": "missing_leaf", "reason": "audit only"})
    with (tmp_path / "input" / "canonical" / "all.json").open("w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False)

    out = tmp_path / "out"
    report = export_sft_dataset(
        tmp_path / "input" / "canonical" / "all.json",
        tmp_path / "input",
        out,
        registry_path,
        config_path,
        corpus=_corpus_map(),
    )
    # id-less non-resolved records are counted, never exported
    assert report["idless_non_resolved_records"] == 2
    assert report["canonical_resolved"] == 3
    assert report["trainable_resolved"] == 3
    rows = pq.read_table(out / "train.parquet").to_pylist()
    assert {row["source_id"] for row in rows} == {"row-train"}


def test_exporter_rejects_resolved_record_without_id(tmp_path):
    canonical = {"row-train": _canonical_record("row-train")}
    canonical["row-train"].pop("id")
    registry_path, config_path, _ = _write_inputs(tmp_path / "input", canonical)

    with pytest.raises(ValueError, match="resolved canonical record without id"):
        export_sft_dataset(
            tmp_path / "input" / "canonical" / "all.json",
            tmp_path / "input",
            tmp_path / "out",
            registry_path,
            config_path,
            corpus=_corpus_map(),
        )


def test_export_report_reports_resolved_outside_splits(tmp_path):
    canonical = {
        "row-train": _canonical_record("row-train", status="resolved"),
        "row-val": _canonical_record("row-val", status="resolved"),
        "row-test": _canonical_record("row-test", status="resolved"),
        "orphan": _canonical_record("orphan", status="resolved"),  # not in any split
    }
    registry_path, config_path, _ = _write_inputs(tmp_path / "input", canonical)
    out = tmp_path / "out"
    report = export_sft_dataset(
        tmp_path / "input" / "canonical" / "all.json",
        tmp_path / "input",
        out,
        registry_path,
        config_path,
        corpus=_corpus_map(),
    )
    assert report["canonical_resolved"] == 4
    assert report["trainable_resolved"] == 3
    assert report["resolved_outside_splits"] == 1
    assert report["resolved_outside_split_ids"] == ["orphan"]


def test_resolved_outside_splits_still_undergo_contract_validation(tmp_path):
    """Contract validation applies to EVERY resolved record, split or not:
    a resolved record outside any split with a category_id absent from the
    registry must fail fast."""
    canonical = {
        "row-train": _canonical_record("row-train", status="resolved"),
        "row-val": _canonical_record("row-val", status="resolved"),
        "row-test": _canonical_record("row-test", status="resolved"),
        # resolved but belongs to no split, and its category_id is invalid
        "orphan": _canonical_record("orphan", category_id="Z", status="resolved"),
    }
    registry_path, config_path, _ = _write_inputs(tmp_path / "input", canonical)

    with pytest.raises(ValueError, match="absent from the leaf registry"):
        export_sft_dataset(
            tmp_path / "input" / "canonical" / "all.json",
            tmp_path / "input",
            tmp_path / "out",
            registry_path,
            config_path,
            corpus=_corpus_map(),
        )


def test_resolved_outside_splits_leaf_name_mismatch_fails_fast(tmp_path):
    canonical = {
        "row-train": _canonical_record("row-train", status="resolved"),
        "row-val": _canonical_record("row-val", status="resolved"),
        "row-test": _canonical_record("row-test", status="resolved"),
        "orphan": _canonical_record("orphan", status="resolved"),
    }
    canonical["orphan"]["target"]["leaf_name"] = "wrong name"
    registry_path, config_path, _ = _write_inputs(tmp_path / "input", canonical)

    with pytest.raises(ValueError, match="does not match registry category"):
        export_sft_dataset(
            tmp_path / "input" / "canonical" / "all.json",
            tmp_path / "input",
            tmp_path / "out",
            registry_path,
            config_path,
            corpus=_corpus_map(),
        )


def test_empty_corpus_is_rejected(tmp_path):
    registry_path, config_path, _ = _write_inputs(tmp_path / "input")

    with pytest.raises(ValueError, match="corpus is required and must be non-empty"):
        export_sft_dataset(
            tmp_path / "input" / "canonical" / "all.json",
            tmp_path / "input",
            tmp_path / "out",
            registry_path,
            config_path,
            corpus={},
        )


def test_incomplete_corpus_is_rejected(tmp_path):
    """Production invariant: the corpus must define every registry category;
    a candidate whose category_id is missing from the corpus must fail fast
    up front instead of late inside prompt building."""
    registry_path, config_path, _ = _write_inputs(tmp_path / "input")
    partial = {category.category_id: category for category in CORPUS[:4]}

    with pytest.raises(ValueError, match="corpus is missing registry categories"):
        export_sft_dataset(
            tmp_path / "input" / "canonical" / "all.json",
            tmp_path / "input",
            tmp_path / "out",
            registry_path,
            config_path,
            corpus=partial,
        )


def test_validate_rejects_empty_corpus(tmp_path):
    registry_path, config_path, _ = _write_inputs(tmp_path / "input")
    out = tmp_path / "out"
    export_sft_dataset(
        tmp_path / "input" / "canonical" / "all.json",
        tmp_path / "input",
        out,
        registry_path,
        config_path,
        corpus=_corpus_map(),
    )

    with pytest.raises(ValueError, match="corpus is required and must be non-empty"):
        validate_sft_dataset(out, registry_path, config_path, corpus={})


def test_classification_is_never_used_as_label_and_is_untouched(tmp_path):
    # canonical record with a level_4 that differs from target.category_id:
    # the target must win (no fallback to classification), and classification
    # itself is never rewritten.
    record = _canonical_record("row-train", category_id="C")
    record["classification"]["level_4"] = "A"
    registry_path, config_path, _ = _write_inputs(
        tmp_path / "input",
        {
            "row-train": record,
            "row-val": _canonical_record("row-val", category_id="C"),
            "row-test": _canonical_record("row-test", category_id="C"),
        },
    )
    out = tmp_path / "out"
    export_sft_dataset(
        tmp_path / "input" / "canonical" / "all.json",
        tmp_path / "input",
        out,
        registry_path,
        config_path,
        corpus=_corpus_map(),
    )
    rows = pq.read_table(out / "train.parquet").to_pylist()
    assert all(row["ground_truth"] == "C" for row in rows)


def test_cli_metadata_fields_override_task_config_and_validator_errors_nonzero(tmp_path):
    registry_path, config_path, corpus_path = _write_inputs(tmp_path / "input")
    config_path.write_text(json.dumps({"task_name": "kept", "metadata_fields": ["table_name"]}), encoding="utf-8")
    out = tmp_path / "out"
    assert export_cli([
        "--canonical", str(tmp_path / "input" / "canonical" / "all.json"),
        "--split-dir", str(tmp_path / "input"),
        "--output-dir", str(out),
        "--registry", str(registry_path),
        "--task-config", str(config_path),
        "--corpus", str(corpus_path),
        "--metadata-fields", "field_name", "field_description",
    ]) == 0
    report = json.loads((out / "export_report.json").read_text(encoding="utf-8"))
    assert report["metadata_fields"] == ["field_name", "field_description"]
    assert report["task_name"] == "kept"
    assert validate_cli([
        "--dataset-dir", str(out), "--registry", str(registry_path),
        "--corpus", str(corpus_path),
        "--metadata-fields", "field_name", "field_description",
    ]) == 0
    (out / "test.parquet").unlink()
    assert validate_cli([
        "--dataset-dir", str(out), "--registry", str(registry_path),
        "--corpus", str(corpus_path),
        "--metadata-fields", "field_name", "field_description",
    ]) == 1


def test_validator_rejects_broken_stage_pairs_and_cross_split_source_ids(tmp_path):
    import pyarrow as pa

    registry_path, config_path, _ = _write_inputs(tmp_path / "input")
    out = tmp_path / "out"
    export_sft_dataset(
        tmp_path / "input" / "canonical" / "all.json",
        tmp_path / "input",
        out,
        registry_path,
        config_path,
        corpus=_corpus_map(),
    )

    train_rows = pq.read_table(out / "train.parquet").to_pylist()
    pq.write_table(pa.Table.from_pylist(train_rows[:1]), out / "train.parquet")
    val_rows = pq.read_table(out / "val.parquet").to_pylist()
    for row in val_rows:
        row["source_id"] = "row-train"
    pq.write_table(pa.Table.from_pylist(val_rows), out / "val.parquet")

    report = validate_sft_dataset(out, registry_path, config_path, corpus=_corpus_map())

    assert report["valid"] is False
    assert any("stage1 and stage2" in error for error in report["splits"]["train"]["errors"])
    assert report["cross_split_errors"]


def test_validator_rejects_prompt_contract_drift(tmp_path):
    import pyarrow as pa

    registry_path, config_path, _ = _write_inputs(tmp_path / "input")
    out = tmp_path / "out"
    export_sft_dataset(
        tmp_path / "input" / "canonical" / "all.json",
        tmp_path / "input",
        out,
        registry_path,
        config_path,
        corpus=_corpus_map(),
    )
    rows = pq.read_table(out / "train.parquet").to_pylist()
    rows[0]["messages"][0]["content"] = "wrong system contract"
    rows[0]["messages"][1]["content"] += "\ntable_name: leaked_secret"
    pq.write_table(pa.Table.from_pylist(rows), out / "train.parquet")

    report = validate_sft_dataset(out, registry_path, config_path, corpus=_corpus_map())

    assert report["valid"] is False
    errors = report["splits"]["train"]["errors"]
    assert any("prompt does not match" in error for error in errors)


def test_validator_rejects_empty_parquet_splits(tmp_path):
    import pyarrow as pa

    registry_path, config_path, _ = _write_inputs(tmp_path / "input")
    out = tmp_path / "out"
    out.mkdir()
    for split in ("train", "val", "test"):
        pq.write_table(pa.table({}), out / f"{split}.parquet")

    report = validate_sft_dataset(out, registry_path, config_path, corpus=_corpus_map())

    assert report["valid"] is False
    assert all(
        any("must contain at least one row" in error for error in details["errors"])
        for details in report["splits"].values()
    )


def test_validator_returns_structured_errors_for_wrong_answers(tmp_path):
    import pyarrow as pa

    registry_path, config_path, _ = _write_inputs(tmp_path / "input")
    out = tmp_path / "out"
    export_sft_dataset(
        tmp_path / "input" / "canonical" / "all.json",
        tmp_path / "input",
        out,
        registry_path,
        config_path,
        corpus=_corpus_map(),
    )
    rows = pq.read_table(out / "train.parquet").to_pylist()
    rows[0]["messages"][-1]["content"] = '{"candidates":["A"]}'
    rows[1]["messages"][-1]["content"] = '{"answer":"A"}'
    pq.write_table(pa.Table.from_pylist(rows), out / "train.parquet")

    report = validate_sft_dataset(out, registry_path, config_path, corpus=_corpus_map())

    assert report["valid"] is False
    errors = report["splits"]["train"]["errors"]
    assert any("stage1 answer" in error for error in errors)
    assert any("equal ground_truth" in error for error in errors)
