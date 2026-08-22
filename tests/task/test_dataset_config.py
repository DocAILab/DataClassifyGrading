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
