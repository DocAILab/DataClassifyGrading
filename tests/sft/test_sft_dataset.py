import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from agent.task import LeafRegistry, TaskConfig
from agent.training.sft import export_sft_dataset, validate_sft_dataset
from script.verl.sft.export import main as export_cli
from script.verl.sft.validate import main as validate_cli


REGISTRY = {
    "categories": [
        {"category_id": "A", "description": "alpha data"},
        {"category_id": "B", "description": "beta data"},
        {"category_id": "C", "description": "gamma data"},
        {"category_id": "D", "description": "delta data"},
        {"category_id": "E", "description": "epsilon data"},
        {"category_id": "F", "description": "zeta data"},
    ]
}
CONFIG = {"metadata_fields": ["field_name", "field_description"]}
RECORD = {
    "id": "row-1",
    "metadata": {"field_name": "email", "field_description": "contact address", "table_name": "secret_table"},
    "classification": {"level_4": "C", "level_1": "", "level_2": "", "level_3": ""},
    "data_level": "L2",
    "label_status": "labeled",
}


def _write_inputs(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        record = {**RECORD, "id": f"row-{split}"}
        (root / f"{split}.json").write_text(json.dumps([record]), encoding="utf-8")
    registry = root / "registry.json"
    registry.write_text(json.dumps(REGISTRY), encoding="utf-8")
    config = root / "task.json"
    config.write_text(json.dumps(CONFIG), encoding="utf-8")
    return registry, config


def test_exporter_writes_messages_parquet_and_report(tmp_path):
    registry_path, config_path = _write_inputs(tmp_path / "input")
    out = tmp_path / "out"
    report = export_sft_dataset(tmp_path / "input", out, registry_path, config_path)
    assert report["format"] == "verl_sft_messages_parquet"
    assert report["splits"]["train"]["exported_records"] == 2
    table = pq.read_table(out / "train.parquet")
    rows = table.to_pylist()
    assert len(rows) == 2
    assert {row["stage"] for row in rows} == {"stage1", "stage2"}
    for row in rows:
        assert [message["role"] for message in row["messages"]] == ["system", "user", "assistant"]
        assert "<|im_start|>" not in json.dumps(row["messages"])
    assert (out / "export_report.json").is_file()
    assert validate_sft_dataset(out, registry_path, config_path)["valid"] is True


def test_candidate_construction_is_deterministic_and_gt_first():
    from agent.training.sft import build_candidates

    registry = LeafRegistry.from_mapping(REGISTRY)
    expected = ["D", "A", "B", "C", "E"]
    assert build_candidates("D", registry) == expected
    assert build_candidates("D", registry) == expected


def test_exporter_rejects_labeled_record_without_leaf_ground_truth(tmp_path):
    registry_path, config_path = _write_inputs(tmp_path / "input")
    bad_record = {**RECORD, "classification": {"level_4": ""}}
    (tmp_path / "input" / "train.json").write_text(
        json.dumps([bad_record]), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="labeled but has no level_4"):
        export_sft_dataset(
            tmp_path / "input", tmp_path / "out", registry_path, config_path
        )


def test_exporter_requires_stable_source_id(tmp_path):
    registry_path, config_path = _write_inputs(tmp_path / "input")
    record_without_id = {key: value for key, value in RECORD.items() if key != "id"}
    (tmp_path / "input" / "train.json").write_text(
        json.dumps([record_without_id]), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="has no stable id"):
        export_sft_dataset(
            tmp_path / "input", tmp_path / "out", registry_path, config_path
        )


def test_cli_metadata_fields_override_task_config_and_validator_errors_nonzero(tmp_path):
    registry_path, config_path = _write_inputs(tmp_path / "input")
    config_path.write_text(json.dumps({"task_name": "kept", "metadata_fields": ["table_name"]}), encoding="utf-8")
    out = tmp_path / "out"
    assert export_cli([
        "--input-dir", str(tmp_path / "input"), "--output-dir", str(out),
        "--registry", str(registry_path), "--task-config", str(config_path),
        "--metadata-fields", "field_name", "field_description",
    ]) == 0
    report = json.loads((out / "export_report.json").read_text(encoding="utf-8"))
    assert report["metadata_fields"] == ["field_name", "field_description"]
    assert report["task_name"] == "kept"
    assert validate_cli([
        "--dataset-dir", str(out), "--registry", str(registry_path),
        "--metadata-fields", "field_name", "field_description",
    ]) == 0
    (out / "test.parquet").unlink()
    assert validate_cli([
        "--dataset-dir", str(out), "--registry", str(registry_path),
        "--metadata-fields", "field_name", "field_description",
    ]) == 1


def test_validator_rejects_broken_stage_pairs_and_cross_split_source_ids(tmp_path):
    import pyarrow as pa

    registry_path, config_path = _write_inputs(tmp_path / "input")
    out = tmp_path / "out"
    export_sft_dataset(tmp_path / "input", out, registry_path, config_path)

    train_rows = pq.read_table(out / "train.parquet").to_pylist()
    pq.write_table(pa.Table.from_pylist(train_rows[:1]), out / "train.parquet")
    val_rows = pq.read_table(out / "val.parquet").to_pylist()
    for row in val_rows:
        row["source_id"] = "row-train"
    pq.write_table(pa.Table.from_pylist(val_rows), out / "val.parquet")

    report = validate_sft_dataset(out, registry_path, config_path)

    assert report["valid"] is False
    assert any("stage1 and stage2" in error for error in report["splits"]["train"]["errors"])
    assert report["cross_split_errors"]


def test_validator_rejects_prompt_contract_drift(tmp_path):
    import pyarrow as pa

    registry_path, config_path = _write_inputs(tmp_path / "input")
    out = tmp_path / "out"
    export_sft_dataset(tmp_path / "input", out, registry_path, config_path)
    rows = pq.read_table(out / "train.parquet").to_pylist()
    rows[0]["messages"][0]["content"] = "wrong system contract"
    rows[0]["messages"][1]["content"] += "\ntable_name: leaked_secret"
    pq.write_table(pa.Table.from_pylist(rows), out / "train.parquet")

    report = validate_sft_dataset(out, registry_path, config_path)

    assert report["valid"] is False
    errors = report["splits"]["train"]["errors"]
    assert any("prompt does not match" in error for error in errors)


def test_validator_rejects_empty_parquet_splits(tmp_path):
    import pyarrow as pa

    registry_path, config_path = _write_inputs(tmp_path / "input")
    out = tmp_path / "out"
    out.mkdir()
    for split in ("train", "val", "test"):
        pq.write_table(pa.table({}), out / f"{split}.parquet")

    report = validate_sft_dataset(out, registry_path, config_path)

    assert report["valid"] is False
    assert all(
        any("must contain at least one row" in error for error in details["errors"])
        for details in report["splits"].values()
    )


def test_validator_returns_structured_errors_for_wrong_answers(tmp_path):
    import pyarrow as pa

    registry_path, config_path = _write_inputs(tmp_path / "input")
    out = tmp_path / "out"
    export_sft_dataset(tmp_path / "input", out, registry_path, config_path)
    rows = pq.read_table(out / "train.parquet").to_pylist()
    rows[0]["messages"][-1]["content"] = '{"candidates":["A"]}'
    rows[1]["messages"][-1]["content"] = '{"answer":"A"}'
    pq.write_table(pa.Table.from_pylist(rows), out / "train.parquet")

    report = validate_sft_dataset(out, registry_path, config_path)

    assert report["valid"] is False
    errors = report["splits"]["train"]["errors"]
    assert any("stage1 answer" in error for error in errors)
    assert any("equal ground_truth" in error for error in errors)
