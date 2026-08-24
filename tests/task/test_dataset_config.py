import json
from pathlib import Path

import pytest

from agent.task import DatasetConfig, load_dataset_configs


ROOT = Path(__file__).resolve().parents[2]


def test_synthetic_dataset_config_loads_from_explicit_path() -> None:
    config = DatasetConfig.from_path(ROOT / "cfg" / "task" / "dataset.example.json")
    assert config.dataset == "demo"
    assert config.path_fields == ("group", "category")
    assert config.leaf_level == "category"


def test_dataset_config_collection_has_no_implicit_builtins(tmp_path: Path) -> None:
    source = tmp_path / "datasets.json"
    source.write_text(
        json.dumps(
            {
                "datasets": {
                    "fixture-one": {
                        "leaf_level": "leaf",
                        "path_fields": ["root", "leaf"],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    configs = load_dataset_configs(source)
    assert tuple(configs) == ("fixture-one",)
    assert configs["fixture-one"].dataset == "fixture-one"


def test_duplicate_dataset_configs_fail(tmp_path: Path) -> None:
    source = tmp_path / "datasets.json"
    source.write_text(
        json.dumps(
            [
                {"dataset": "demo", "path_fields": ["level_4"]},
                {"dataset": "demo", "path_fields": ["level_4"]},
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate dataset config"):
        load_dataset_configs(source)


def test_registry_derivation_and_projection_fields_round_trip(tmp_path) -> None:
    from agent.task import REGISTRY_DERIVATIONS

    source = tmp_path / "datasets.json"
    source.write_text(
        json.dumps(
            {
                "datasets": {
                    "demo_coded": {
                        "leaf_level": "level_4",
                        "id_strategy": "code",
                        "path_fields": ["level_1", "level_2", "level_3", "level_4"],
                        "placeholder_labels": ["--"],
                        "projection_excluded_category_ids": ["Z9-9"],
                        "registry_source": "demo_other",
                        "registry_derivation": "shared-standard",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    config = load_dataset_configs(source)["demo_coded"]
    assert config.projection_excluded_category_ids == ("Z9-9",)
    assert config.registry_derivation == "shared-standard"
    mapping = config.to_mapping()
    assert mapping["projection_excluded_category_ids"] == ["Z9-9"]
    assert mapping["registry_derivation"] == "shared-standard"
    assert set(REGISTRY_DERIVATIONS) == {
        "standard",
        "shared-standard",
        "dataset-universe",
    }


def test_invalid_registry_derivation_rejected() -> None:
    with pytest.raises(ValueError, match="registry_derivation"):
        DatasetConfig(dataset="demo", registry_derivation="vibes")


def test_shared_standard_requires_registry_source() -> None:
    with pytest.raises(ValueError, match="shared-standard.*requires"):
        DatasetConfig(dataset="demo", registry_derivation="shared-standard")


def test_duplicate_projection_exclusions_rejected() -> None:
    with pytest.raises(ValueError, match="must be unique"):
        DatasetConfig(
            dataset="demo",
            projection_excluded_category_ids=("A", "A"),
        )
