from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.task.dcg_shougang import PATH_FIELDS, prepare_shougang_5415
from agent.task.identity import qualified_category_id, stable_record_id


def _row(index: int, leaf: str, *, level: str = "L2", source_id: str | None = None) -> dict:
    return {
        "id": source_id or f"source-{index}",
        "key": f"field-{index}",
        "label_status": "labeled",
        "metadata": {
            "database_name": "steel_db",
            "database_description": "steel database",
            "table_name": f"table_{index}",
            "table_description": f"table description {index}",
            "field_name": f"field_{index}",
            "field_description": f"field description {index}",
            "field_type": "varchar",
            "value": "",
        },
        "classification": {
            "level_1": "production",
            "level_2": "quality",
            "level_3": "inspection",
            "level_4": leaf,
        },
        "data_level": level,
    }


def _write_split(root: Path, split: str, rows: list[dict]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{split}.json").write_text(
        json.dumps(rows, ensure_ascii=False), encoding="utf-8"
    )


def _five_category_source(tmp_path: Path) -> Path:
    source = tmp_path / "shougang-5415"
    train = [_row(index, f"leaf-{index}", level=f"L{(index % 4) + 1}") for index in range(5)]
    _write_split(source, "train", train)
    _write_split(source, "val", [_row(10, "leaf-0", level="L1")])
    _write_split(source, "test", [_row(11, "leaf-1", level="L2")])
    return source


def test_prepare_preserves_official_splits_and_builds_assets(tmp_path: Path) -> None:
    source = _five_category_source(tmp_path)

    prepared = prepare_shougang_5415(source, strict_counts=False)

    assert prepared.report["split_counts"] == {"train": 5, "val": 1, "test": 1}
    assert prepared.report["total_records"] == 7
    assert prepared.report["category_count"] == 5
    assert [row["split"] for row in prepared.canonical_records] == [
        "train", "train", "train", "train", "train", "val", "test"
    ]

    first = prepared.canonical_records[0]
    expected_path = ("production", "quality", "inspection", "leaf-0")
    expected_category_id = qualified_category_id("shougang", ("leaf-0",))
    assert first["source_id"] == "source-0"
    assert first["id"] == stable_record_id("shougang", first["metadata"])
    assert first["schema_version"] == 2
    assert first["resolution_status"] == "resolved"
    assert first["target"] == {
        "leaf_level": "level_4",
        "leaf_name": "leaf-0",
        "category_id": expected_category_id,
        "category_path": list(expected_path),
    }

    categories = prepared.registry["categories"]
    assert len(categories) == 5
    assert categories == sorted(categories, key=lambda item: item["category_id"])
    assert prepared.corpus["categories"][0]["description"].startswith(
        "Derived taxonomy path: "
    )
    assert prepared.dataset_config == {
        "datasets": {
            "shougang": {
                "leaf_level": "level_4",
                "id_strategy": "path",
                "path_fields": ["level_1", "level_2", "level_3", "level_4"],
                "identity_fields": ["level_4"],
                "registry_derivation": "dataset-universe",
            }
        }
    }
    assert prepared.task_config["metadata_fields"] == [
        "field_name", "table_name", "field_description", "table_description"
    ]
    assert prepared.grading_config == {
        "levels": ["L1", "L2", "L3", "L4"],
        "gt_field": "data_level",
    }


def test_prepare_output_is_deterministic(tmp_path: Path) -> None:
    source = _five_category_source(tmp_path)
    first = prepare_shougang_5415(source, strict_counts=False)
    second = prepare_shougang_5415(source, strict_counts=False)

    assert first.to_mapping() == second.to_mapping()


def test_prepare_supports_full_taxonomy_path_identity(tmp_path: Path) -> None:
    source = _five_category_source(tmp_path)
    rows = json.loads((source / "train.json").read_text(encoding="utf-8"))
    alternate = _row(20, "leaf-1")
    alternate["classification"]["level_1"] = "management"
    alternate["classification"]["level_2"] = "planning"
    alternate["classification"]["level_3"] = "dispatch"
    rows.append(alternate)
    _write_split(source, "train", rows)

    prepared = prepare_shougang_5415(
        source,
        strict_counts=False,
        identity_fields=PATH_FIELDS,
    )

    assert prepared.report["category_count"] == 6
    assert prepared.dataset_config["datasets"]["shougang"]["identity_fields"] == list(
        PATH_FIELDS
    )
    category_ids = {
        row["target"]["category_id"]
        for row in prepared.canonical_records
    }
    assert qualified_category_id(
        "shougang", ("management", "planning", "dispatch", "leaf-1")
    ) in category_ids


def test_same_leaf_under_multiple_parent_paths_remains_one_official_class(
    tmp_path: Path,
) -> None:
    source = _five_category_source(tmp_path)
    train_path = source / "train.json"
    rows = json.loads(train_path.read_text(encoding="utf-8"))
    duplicate_leaf = _row(20, "leaf-0")
    duplicate_leaf["classification"]["level_2"] = "planning"
    rows.append(duplicate_leaf)
    _write_split(source, "train", rows)

    prepared = prepare_shougang_5415(source, strict_counts=False)

    assert prepared.report["category_count"] == 5
    assert prepared.report["taxonomy_path_count"] == 6
    assert prepared.report["multi_path_category_count"] == 1
    matching = [
        item for item in prepared.corpus["categories"] if item["name"] == "leaf-0"
    ]
    assert len(matching) == 1
    assert len(matching[0]["descriptions"]) == 1
    assert {
        row["target"]["category_id"]
        for row in prepared.canonical_records
        if row["target"]["leaf_name"] == "leaf-0"
    } == {qualified_category_id("shougang", ("leaf-0",))}


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda rows: rows[0].update(data_level="L5"), "data_level"),
        (
            lambda rows: rows[0]["classification"].update(level_3=""),
            "complete taxonomy path",
        ),
    ],
)
def test_prepare_rejects_invalid_labels(
    tmp_path: Path, mutation, message: str
) -> None:
    source = _five_category_source(tmp_path)
    train_path = source / "train.json"
    rows = json.loads(train_path.read_text(encoding="utf-8"))
    mutation(rows)
    _write_split(source, "train", rows)

    with pytest.raises(ValueError, match=message):
        prepare_shougang_5415(source, strict_counts=False)


def test_prepare_rejects_duplicate_source_ids_across_splits(tmp_path: Path) -> None:
    source = _five_category_source(tmp_path)
    _write_split(source, "test", [_row(11, "leaf-1", source_id="source-0")])

    with pytest.raises(ValueError, match="duplicate source id"):
        prepare_shougang_5415(source, strict_counts=False)


def test_prepare_rejects_evaluation_category_absent_from_train(tmp_path: Path) -> None:
    source = _five_category_source(tmp_path)
    _write_split(source, "test", [_row(11, "unseen-leaf")])

    with pytest.raises(ValueError, match="absent from train"):
        prepare_shougang_5415(source, strict_counts=False)


def test_prepare_strict_mode_enforces_official_contract(tmp_path: Path) -> None:
    source = _five_category_source(tmp_path)

    with pytest.raises(ValueError, match="split counts"):
        prepare_shougang_5415(source)


def test_cli_publishes_complete_runtime_layout(tmp_path: Path) -> None:
    from script.canonical.import_dcg_shougang import main

    source = _five_category_source(tmp_path)
    output = tmp_path / "runtime"

    assert main([
        "--source-dir", str(source),
        "--output-dir", str(output),
        "--no-strict-counts",
    ]) == 0

    expected = {
        "canonical/shougang/all.json",
        "canonical/shougang/import_report.json",
        "registries/shougang.registry.json",
        "corpus/shougang.corpus.json",
        "cfg/datasets.json",
        "cfg/task.json",
        "cfg/grading.json",
    }
    actual = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file()
    }
    assert actual == expected
    published = json.loads(
        (output / "canonical/shougang/all.json").read_text(encoding="utf-8")
    )
    assert len(published) == 7


def test_cli_can_publish_full_path_identity_release(tmp_path: Path) -> None:
    from script.canonical.import_dcg_shougang import main

    source = _five_category_source(tmp_path)
    output = tmp_path / "runtime-full-path"

    assert main([
        "--source-dir", str(source),
        "--output-dir", str(output),
        "--no-strict-counts",
        "--identity-mode", "full-path",
    ]) == 0

    config = json.loads(
        (output / "cfg/datasets.json").read_text(encoding="utf-8")
    )
    assert config["datasets"]["shougang"]["identity_fields"] == list(PATH_FIELDS)


def test_cli_refuses_nonempty_output_without_overwrite(tmp_path: Path) -> None:
    from script.canonical.import_dcg_shougang import main

    source = _five_category_source(tmp_path)
    output = tmp_path / "runtime"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("user data", encoding="utf-8")

    with pytest.raises(FileExistsError, match="non-empty"):
        main([
            "--source-dir", str(source),
            "--output-dir", str(output),
            "--no-strict-counts",
        ])
    assert sentinel.read_text(encoding="utf-8") == "user data"


def test_cli_invalid_input_leaves_no_partial_release(tmp_path: Path) -> None:
    from script.canonical.import_dcg_shougang import main

    source = _five_category_source(tmp_path)
    (source / "test.json").unlink()
    output = tmp_path / "runtime"

    with pytest.raises(FileNotFoundError, match="test.json"):
        main([
            "--source-dir", str(source),
            "--output-dir", str(output),
            "--no-strict-counts",
        ])
    assert not output.exists()
