"""Synthetic end-to-end tests for the canonical builder (schema v2).

Fixture shapes mirror the audited real-data hazards without carrying any
real production values:
- full four-level paths and paths with an empty level_3;
- the same leaf name under two different parent paths (must yield distinct
  category_ids);
- a dash-like placeholder leaf appearing under many parent paths;
- a truncated/malformed leaf label that must stay unresolved with its raw
  text visible in the reason;
- code-strategy resolution driven by a corpus code map;
- projection-excluded category hits failing fast.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.task import (
    CanonicalBuildResult,
    DatasetConfig,
    LeafRegistry,
    load_dataset_configs,
)
from agent.task.canonical_builder import (
    load_corpus_categories_file,
    prepare_canonical_dataset,
    resolve_record,
    write_canonical_dataset,
)
from agent.task.contracts import CorpusCategory
from agent.task.dataset_config import REGISTRY_DERIVATIONS
from agent.task.identity import stable_record_id
from agent.task.resolver import ClassificationTargetResolver


ROOT = Path(__file__).resolve().parents[2]


def _category(category_id: str, name: str, path: list[str]) -> dict:
    return {"category_id": category_id, "name": name, "path": path}


@pytest.fixture()
def path_registry() -> LeafRegistry:
    """Six categories; 'D1' deliberately appears under two parents."""
    return LeafRegistry.from_mapping(
        [
            _category("demo_four:A.B.C.D1", "D1", ["A", "B", "C", "D1"]),
            _category("demo_four:A.B.D1", "D1", ["A", "B", "D1"]),
            _category("demo_four:E.F.G.H1", "H1", ["E", "F", "G", "H1"]),
            _category("demo_four:E2.F2.G2.H2", "H2", ["E2", "F2", "G2", "H2"]),
            _category("demo_four:S1", "S1", ["S1"]),
            _category("demo_four:S2", "S2", ["S2"]),
        ]
    )


def _record(
    record_id: str,
    levels: dict[str, str],
    *,
    label_status: str = "labeled",
    table: str = "T1",
    dataset: str = "demo_four",
) -> dict:
    metadata = {
        "database_name": "DB",
        "table_name": table,
        "field_name": f"F_{record_id}",
        "field_description": "",
        "field_type": "STRING",
    }
    return {
        "id": stable_record_id(dataset, metadata),
        "key": record_id,
        "label_status": label_status,
        "metadata": metadata,
        "classification": {
            "level_1": levels.get("level_1", ""),
            "level_2": levels.get("level_2", ""),
            "level_3": levels.get("level_3", ""),
            "level_4": levels.get("level_4", ""),
        },
        "data_level": "L2",
    }


FOURLEVEL = dict(
    dataset="demo_four",
    leaf_level="level_4",
    id_strategy="path",
    path_fields=("level_1", "level_2", "level_3", "level_4"),
)


# ---------------------------------------------------------------------------
# resolution outcomes
# ---------------------------------------------------------------------------


def test_canonical_rejects_stale_record_id(path_registry) -> None:
    config = DatasetConfig(**FOURLEVEL)
    record = _record("stale", {"level_1": "A", "level_2": "B", "level_3": "C", "level_4": "D1"})
    record["id"] = "stale-id"
    with pytest.raises(ValueError, match="stale.*id"):
        prepare_canonical_dataset(
            "demo_four",
            processed_file=_write_records([record]),
            output_file="out/all.json",
            config=config,
            registry=path_registry,
        )


def test_canonical_rejects_duplicate_stable_record_ids(path_registry) -> None:
    config = DatasetConfig(**FOURLEVEL)
    record = _record(
        "dup",
        {"level_1": "A", "level_2": "B", "level_3": "C", "level_4": "D1"},
    )
    duplicate = json.loads(json.dumps(record))
    duplicate["key"] = "another-row-view"
    with pytest.raises(ValueError, match="duplicate stable record id"):
        prepare_canonical_dataset(
            "demo_four",
            processed_file=_write_records([record, duplicate]),
            output_file="out/all.json",
            config=config,
            registry=path_registry,
        )


def test_resolved_full_path_and_v2_enrichment(path_registry) -> None:
    config = DatasetConfig(**FOURLEVEL)
    record = _record("r1", {"level_1": "A", "level_2": "B", "level_3": "C", "level_4": "D1"})
    result = resolve_record(record, ClassificationTargetResolver(config), path_registry, {"D1", "H1", "H2", "S1", "S2"})
    assert result.status.value == "resolved"
    assert result.target.category_id == "demo_four:A.B.C.D1"

    build_result, records = prepare_canonical_dataset(
        "demo_four",
        processed_file=_write_records([record]),
        output_file="out/all.json",
        config=config,
        registry=path_registry,
        registry_file="registries/demo.registry.json",
    )
    row = records[0]
    assert row["schema_version"] == 2
    assert row["dataset"] == "demo_four"
    assert row["path_mask"] == [True, True, True, True]
    assert row["leaf"] == {"field": "level_4", "value": "D1"}
    assert row["resolution"] == {
        "status": "resolved",
        "category_id": "demo_four:A.B.C.D1",
        "reason": "",
    }
    # v1 compatibility keys stay aligned
    assert row["resolution_status"] == row["resolution"]["status"]
    assert row["target"]["category_id"] == "demo_four:A.B.C.D1"
    assert row["split"] is None and row["split_exclusion_reason"] is None
    # input fields are passed through untouched
    assert row["metadata"]["table_name"] == "T1"
    assert build_result.input_records == build_result.output_records == 1


def _write_records(records: list[dict]) -> str:
    import tempfile

    handle = tempfile.NamedTemporaryFile(
        "w", suffix=".json", delete=False, encoding="utf-8", newline="\n"
    )
    json.dump(records, handle, ensure_ascii=False)
    handle.close()
    return handle.name


def test_empty_level_kept_explicit_and_identity_projection(path_registry) -> None:
    config = DatasetConfig(**{**FOURLEVEL, "identity_fields": ("level_1", "level_2", "level_4")})
    names = {"D1", "H1", "H2", "S1", "S2"}
    record = _record("p1", {"level_1": "A", "level_2": "B", "level_3": "", "level_4": "D1"})
    result = resolve_record(record, ClassificationTargetResolver(config), path_registry, names)
    assert result.status.value == "resolved"
    # projected identity skips the empty level but stays distinguishable
    assert result.target.category_id == "demo_four:A.B.D1"
    _, records = prepare_canonical_dataset(
        "demo_four",
        processed_file=_write_records([record]),
        output_file="out/all.json",
        config=config,
        registry=path_registry,
    )
    assert records[0]["path_mask"] == [True, True, False, True]


def test_same_leaf_name_under_different_parents_distinct_ids(path_registry) -> None:
    config = DatasetConfig(**FOURLEVEL)
    resolver = ClassificationTargetResolver(config)
    names = {"D1", "H1", "H2", "S1", "S2"}
    first = resolve_record(
        _record("a", {"level_1": "A", "level_2": "B", "level_3": "C", "level_4": "D1"}),
        resolver, path_registry, names,
    )
    projected = ClassificationTargetResolver(
        DatasetConfig(**{**FOURLEVEL, "identity_fields": ("level_1", "level_2", "level_4")})
    )
    second = resolve_record(
        _record("b", {"level_1": "A", "level_2": "B", "level_3": "", "level_4": "D1"}),
        projected,
        path_registry,
        names,
    )
    assert first.target.category_id != second.target.category_id
    assert first.target.leaf_name == second.target.leaf_name == "D1"


def test_placeholder_dash_under_many_paths_excluded(path_registry) -> None:
    config = DatasetConfig(**{**FOURLEVEL, "placeholder_labels": ("——",)})
    names = {"D1", "H1", "H2", "S1", "S2"}
    records = [
        _record(f"d{i}", {"level_1": dom, "level_2": "P", "level_3": "S", "level_4": "——"})
        for i, dom in enumerate(["生产A", "生产B", "管理C"])
    ]
    result, out = prepare_canonical_dataset(
        "demo_four",
        processed_file=_write_records(records),
        output_file="out/all.json",
        config=config,
        registry=path_registry,
    )
    assert result.status_counts == {"placeholder": 3}
    assert all(row["resolution"]["status"] == "placeholder" for row in out)
    assert all("target" not in row for row in out)
    assert result.unresolved_details["missing_leaf"]["count"] == 0


def test_malformed_label_stays_missing_leaf_with_raw_text(path_registry) -> None:
    config = DatasetConfig(**FOURLEVEL)
    raw_leaf = "Basic Info (Public"  # truncated-parenthesis style hazard
    record = _record("m1", {"level_1": "A", "level_2": "B", "level_3": "C", "level_4": raw_leaf})
    result, out = prepare_canonical_dataset(
        "demo_four",
        processed_file=_write_records([record]),
        output_file="out/all.json",
        config=config,
        registry=path_registry,
    )
    assert result.status_counts == {"missing_leaf": 1}
    detail = result.unresolved_details["missing_leaf"]
    assert detail["by_leaf"] == {raw_leaf: 1}
    assert raw_leaf in out[0]["resolution"]["reason"]
    assert out[0]["leaf"]["value"] == raw_leaf
    assert "target" not in out[0]


def test_unlabeled_beats_leaf_presence() -> None:
    config = DatasetConfig(dataset="demo_single", leaf_level="level_4", path_fields=("level_4",))
    record = _record(
        "u1", {"level_4": "S1"}, label_status="unlabeled", dataset="demo_single"
    )
    result, out = prepare_canonical_dataset(
        "demo_single",
        processed_file=_write_records([record]),
        output_file="out/all.json",
        config=config,
        registry=LeafRegistry.from_mapping(
            [_category("demo:S1", "S1", ["S1"]), _category("demo:S2", "S2", ["S2"])]
            + [_category(f"demo:X{i}", f"X{i}", []) for i in range(3)]
        ),
    )
    assert result.status_counts == {"unlabeled": 1}
    assert out[0]["leaf"]["value"] == "S1"
    assert "target" not in out[0]


def test_non_mapping_record_wrapped_not_fatal(path_registry) -> None:
    config = DatasetConfig(**FOURLEVEL)
    result, out = prepare_canonical_dataset(
        "demo_four",
        processed_file=_write_records(["just-a-string"]),
        output_file="out/all.json",
        config=config,
        registry=path_registry,
    )
    assert result.status_counts == {"invalid_record": 1}
    assert out[0]["record"] == "just-a-string"
    assert out[0]["path_mask"] == [False, False, False, False]


# ---------------------------------------------------------------------------
# code strategy
# ---------------------------------------------------------------------------


def _code_assets():
    categories = [
        CorpusCategory(category_id=f"C-{i}", name=name, code=f"C-{i}")
        for i, name in enumerate(["K1", "K2", "K3", "K4", "K5"], start=1)
    ]
    registry = LeafRegistry.from_mapping(
        [{"category_id": f"C-{i}", "name": name, "code": f"C-{i}"}
         for i, name in enumerate(["K1", "K2", "K3", "K4", "K5"], start=1)]
    )
    config = DatasetConfig(
        dataset="demo_coded",
        leaf_level="level_4",
        id_strategy="code",
        path_fields=("level_4",),
        placeholder_labels=("--",),
    )
    return categories, registry, config


def test_code_strategy_resolves_via_corpus_map() -> None:
    corpus, registry, config = _code_assets()
    records = [
        _record("c1", {"level_4": "K1"}, dataset="demo_coded"),
        _record("c2", {"level_4": "--"}, dataset="demo_coded"),
        _record("c3", {"level_4": "UNKNOWN"}, dataset="demo_coded"),
    ]
    result, out = prepare_canonical_dataset(
        "demo_coded",
        processed_file=_write_records(records),
        output_file="out/all.json",
        config=config,
        registry=registry,
        corpus_categories=corpus,
    )
    assert result.status_counts == {"code_unresolved": 1, "placeholder": 1, "resolved": 1}
    by_key = {row["key"]: row for row in out}
    assert by_key["c1"]["target"]["category_id"] == "C-1"
    assert by_key["c3"]["resolution"]["category_id"] is None
    assert "UNKNOWN" in by_key["c3"]["resolution"]["reason"]
    assert result.unresolved_details["code_unresolved"]["by_leaf"] == {"UNKNOWN": 1}


def test_code_strategy_requires_corpus() -> None:
    _, _, config = _code_assets()
    with pytest.raises(ValueError, match="requires corpus"):
        prepare_canonical_dataset(
            "demo_coded",
            processed_file=_write_records(
                [_record("c1", {"level_4": "K1"}, dataset="demo_coded")]
            ),
            output_file="out/all.json",
            config=config,
            registry=LeafRegistry.from_mapping(
                [_category(f"C-{i}", f"K{i}", []) for i in range(1, 6)]
            ),
        )


# ---------------------------------------------------------------------------
# guards / determinism / persistence
# ---------------------------------------------------------------------------


def test_projection_excluded_hit_fails_fast(path_registry) -> None:
    config = DatasetConfig(
        **{**FOURLEVEL, "projection_excluded_category_ids": ("demo_four:E.F.G.H1",)}
    )
    record = _record("x1", {"level_1": "E", "level_2": "F", "level_3": "G", "level_4": "H1"})
    with pytest.raises(ValueError, match="projection-excluded"):
        resolve_record(
            record,
            ClassificationTargetResolver(config),
            path_registry,
            {"D1", "H1", "H2", "S1", "S2"},
            excluded_ids=frozenset(config.projection_excluded_category_ids),
        )


def test_build_is_byte_identical_on_rerun(tmp_path, path_registry) -> None:
    config = DatasetConfig(**FOURLEVEL)
    records = [
        _record("r1", {"level_1": "A", "level_2": "B", "level_3": "C", "level_4": "D1"}),
        _record("r2", {"level_1": "E", "level_2": "F", "level_3": "G", "level_4": "H1"}),
    ]
    processed = Path(_write_records(records))

    def run(destination: Path) -> None:
        result, rows = prepare_canonical_dataset(
            "demo_four",
            processed_file=processed,
            output_file=destination / "all.json",
            config=config,
            registry=path_registry,
            registry_file="registries/demo.registry.json",
        )
        write_canonical_dataset(result, rows, overwrite=True)

    first, second = tmp_path / "one", tmp_path / "two"
    run(first)
    run(second)
    assert (first / "all.json").read_bytes() == (second / "all.json").read_bytes()
    one = json.loads((first / "resolution_report.json").read_text(encoding="utf-8"))
    two = json.loads((second / "resolution_report.json").read_text(encoding="utf-8"))
    assert one.pop("output_file") != two.pop("output_file")  # caller paths differ by design
    assert one == two


def test_write_refuses_overwrite_without_flag(tmp_path, path_registry) -> None:
    config = DatasetConfig(**FOURLEVEL)
    result, rows = prepare_canonical_dataset(
        "demo_four",
        processed_file=_write_records(
            [_record("r1", {"level_1": "A", "level_2": "B", "level_3": "C", "level_4": "D1"})]
        ),
        output_file=tmp_path / "all.json",
        config=config,
        registry=path_registry,
    )
    write_canonical_dataset(result, rows)
    with pytest.raises(FileExistsError):
        write_canonical_dataset(result, rows)


def test_example_config_file_loads_with_all_shapes() -> None:
    configs = load_dataset_configs(ROOT / "cfg" / "task" / "datasets.example.json")
    assert set(configs) == {"demo_fourlevel", "demo_partial", "demo_single", "demo_shared"}
    assert configs["demo_partial"].identity_fields == ("level_1", "level_2", "level_4")
    assert configs["demo_single"].registry_derivation == "dataset-universe"
    shared = configs["demo_shared"]
    assert shared.registry_source == "demo_fourlevel"
    assert shared.id_strategy == "code"
    assert shared.projection_excluded_category_ids == ("Z9-9",)
    assert set(REGISTRY_DERIVATIONS) >= {"standard", "shared-standard", "dataset-universe"}


def test_cli_builds_canonical_layer(tmp_path, path_registry) -> None:
    from script.canonical.build import main

    processed = tmp_path / "processed" / "demo_four"
    processed.mkdir(parents=True)
    (processed / "all.json").write_text(
        json.dumps(
            [_record("r1", {"level_1": "A", "level_2": "B", "level_3": "C", "level_4": "D1"})]
        ),
        encoding="utf-8",
    )
    config_file = tmp_path / "configs.json"
    config_file.write_text(
        json.dumps({"datasets": {"demo_four": {**FOURLEVEL}}}),
        encoding="utf-8",
    )
    registry_dir = tmp_path / "registries"
    registry_dir.mkdir()
    (registry_dir / "demo_four.registry.json").write_text(
        json.dumps({"categories": [
            _category("demo_four:A.B.C.D1", "D1", ["A", "B", "C", "D1"]),
            _category("demo_four:A.B.D1", "D1", ["A", "B", "D1"]),
            _category("demo_four:E.F.G.H1", "H1", ["E", "F", "G", "H1"]),
            _category("demo_four:E2.F2.G2.H2", "H2", ["E2", "F2", "G2", "H2"]),
            _category("demo:S1", "S1", ["S1"]),
        ]}, ensure_ascii=False),
        encoding="utf-8",
    )

    common = [
        "--processed-dir", str(tmp_path / "processed"),
        "--canonical-dir", str(tmp_path / "canonical"),
        "--config-file", str(config_file),
        "--registry-dir", str(registry_dir),
        "--dataset", "demo_four",
    ]
    assert main([*common, "--overwrite"]) == 0
    canonical = tmp_path / "canonical" / "demo_four"
    rows = json.loads((canonical / "all.json").read_text(encoding="utf-8"))
    assert rows[0]["resolution"]["status"] == "resolved"
    report = json.loads((canonical / "resolution_report.json").read_text(encoding="utf-8"))
    assert report["schema_version"] == 2
    assert report["input_sha256"]

    # second run without --overwrite refuses
    with pytest.raises(FileExistsError):
        main(common)

    # unknown dataset fails fast
    with pytest.raises(SystemExit):
        main([*common, "--overwrite", "--dataset", "nope"])
