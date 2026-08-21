import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from agent.task import LeafRegistry, TaskConfig
from method.sft import export_sft_dataset, validate_sft_dataset
from method.sft.dataset import build_random_shuffled_candidates, load_splits
from method.sft.script.export import main as export_cli
from method.sft.script.validate import main as validate_cli


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
    from method.sft import build_candidates

    registry = LeafRegistry.from_mapping(REGISTRY)
    expected = ["D", "A", "B", "C", "E"]
    assert build_candidates("D", registry) == expected
    assert build_candidates("D", registry) == expected


def test_random_shuffled_candidates_are_valid_and_deterministic():
    registry = LeafRegistry.from_mapping(REGISTRY)

    first = build_random_shuffled_candidates(
        "A", registry, source_id="row-17", seed=42
    )
    second = build_random_shuffled_candidates(
        "A", registry, source_id="row-17", seed=42
    )

    assert first == second
    assert len(first) == len(set(first)) == 5
    assert first.count("A") == 1
    assert set(first) <= set(registry.ids)
    assert build_random_shuffled_candidates(
        "A", registry, source_id="row-17", seed=137
    ) != first


def test_random_shuffled_candidates_do_not_fix_golden_position():
    registry = LeafRegistry.from_mapping(REGISTRY)

    positions = {
        build_random_shuffled_candidates(
            "A", registry, source_id=f"row-{index}", seed=42
        ).index("A")
        for index in range(100)
    }

    assert positions == {0, 1, 2, 3, 4}


@pytest.mark.parametrize("source_id", ["", "   "])
def test_random_shuffled_candidates_require_source_id(source_id):
    registry = LeafRegistry.from_mapping(REGISTRY)

    with pytest.raises(ValueError, match="source_id"):
        build_random_shuffled_candidates(
            "A", registry, source_id=source_id, seed=42
        )


def test_random_shuffled_candidates_reject_oov_golden():
    registry = LeafRegistry.from_mapping(REGISTRY)

    with pytest.raises(ValueError, match="ground truth"):
        build_random_shuffled_candidates(
            "outside", registry, source_id="row-1", seed=42
        )


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
    (out / "val.parquet").unlink()
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


def test_load_splits_explicit_train_val_never_opens_test(tmp_path):
    (tmp_path / "train.json").write_text("[]", encoding="utf-8")
    (tmp_path / "val.json").write_text("[]", encoding="utf-8")
    (tmp_path / "test.json").write_text("not-json", encoding="utf-8")

    assert load_splits(tmp_path, ("train", "val")) == {"train": [], "val": []}


def test_load_splits_rejects_test_before_file_access(tmp_path):
    (tmp_path / "test.json").write_text("not-json", encoding="utf-8")

    with pytest.raises(ValueError, match="test"):
        load_splits(tmp_path, ("test",))


def test_field_name_only_export_reports_input_and_external_corpus_contract(tmp_path):
    registry_path, config_path = _write_inputs(tmp_path / "input")
    config_path.write_text(
        json.dumps({"task_name": "field_name_level4", "metadata_fields": ["field_name"]}),
        encoding="utf-8",
    )

    out = tmp_path / "out"
    report = export_sft_dataset(
        tmp_path / "input", out, registry_path, config_path, splits=("train", "val")
    )
    rows = pq.read_table(out / "train.parquet").to_pylist()

    assert report["requested_splits"] == ["train", "val"]
    assert report["real_test_split_read"] is False
    assert report["metadata_fields"] == ["field_name"]
    assert report["supervision_target"] == "classification.level_4"
    assert report["external_corpus"] == "leaf_registry_descriptions"
    assert all(row["metadata"] == {"field_name": "email"} for row in rows)
    serialized = json.dumps(rows, ensure_ascii=False)
    assert "contact address" not in serialized
    assert "secret_table" not in serialized


def test_export_random_shuffled_policy_records_candidate_audit(tmp_path):
    registry_path, config_path = _write_inputs(tmp_path / "input")
    out = tmp_path / "out"

    report = export_sft_dataset(
        tmp_path / "input",
        out,
        registry_path,
        config_path,
        splits=("train",),
        candidate_policy="random-shuffled",
        candidate_seed=42,
    )
    rows = pq.read_table(out / "train.parquet").to_pylist()

    assert report["candidate_policy"] == "random-shuffled"
    assert report["candidate_seed"] == 42
    assert report["candidate_policy_version"] == "random_shuffled_v1"
    assert report["real_test_split_read"] is False
    assert report["candidate_duplicate_rows"] == 0
    assert report["candidate_oov_rows"] == 0
    assert sum(report["golden_position_histogram"].values()) == 1
    assert rows[0]["candidates"] == rows[1]["candidates"]
    assert all(row["candidate_policy"] == "random-shuffled" for row in rows)
    assert all(row["candidate_seed"] == 42 for row in rows)
    assert all(row["golden_position"] == row["candidates"].index("C") for row in rows)


def test_export_defaults_to_random_shuffled_candidates(tmp_path):
    registry_path, config_path = _write_inputs(tmp_path / "input")
    out = tmp_path / "out"

    report = export_sft_dataset(
        tmp_path / "input",
        out,
        registry_path,
        config_path,
        splits=("train",),
    )

    assert report["candidate_policy"] == "random-shuffled"
    assert report["candidate_seed"] == 42
    assert report["candidate_policy_version"] == "random_shuffled_v1"


def test_export_cli_defaults_to_random_shuffled_candidates(tmp_path):
    registry_path, config_path = _write_inputs(tmp_path / "input")
    out = tmp_path / "out"

    return_code = export_cli(
        [
            "--input-dir", str(tmp_path / "input"),
            "--output-dir", str(out),
            "--registry", str(registry_path),
            "--task-config", str(config_path),
            "--metadata-fields", "field_name",
            "--splits", "train",
        ]
    )

    assert return_code == 0
    report = json.loads((out / "export_report.json").read_text(encoding="utf-8"))
    assert report["candidate_policy"] == "random-shuffled"
    assert report["candidate_seed"] == 42


def test_export_cli_selects_random_shuffled_policy(tmp_path):
    registry_path, config_path = _write_inputs(tmp_path / "input")
    out = tmp_path / "out"

    return_code = export_cli(
        [
            "--input-dir", str(tmp_path / "input"),
            "--output-dir", str(out),
            "--registry", str(registry_path),
            "--task-config", str(config_path),
            "--metadata-fields", "field_name",
            "--splits", "train",
            "--candidate-policy", "random-shuffled",
            "--candidate-seed", "42",
        ]
    )

    assert return_code == 0
    report = json.loads((out / "export_report.json").read_text(encoding="utf-8"))
    assert report["candidate_policy"] == "random-shuffled"
    assert report["candidate_seed"] == 42
    assert report["requested_splits"] == ["train"]
    assert report["real_test_split_read"] is False
