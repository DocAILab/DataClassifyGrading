"""Per-dataset grading-standard manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from agent.task import GradingConfig
from agent.task.grading_manifest import DatasetGradingManifest


def _write_grading(path: Path, prefix: str) -> str:
    path.write_text(
        json.dumps(
            {
                "levels": ["L1", "L2", "L3", "L4"],
                "descriptions": [f"{prefix}-{index}" for index in range(1, 5)],
                "gt_field": "data_level",
            }
        ),
        encoding="utf-8",
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_manifest_loads_and_verifies_each_dataset_standard(tmp_path: Path) -> None:
    finance = tmp_path / "finance.json"
    shougang = tmp_path / "shougang.json"
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "datasets": {
                    "finance": {"path": finance.name, "sha256": _write_grading(finance, "F")},
                    "shougang": {"path": shougang.name, "sha256": _write_grading(shougang, "S")},
                }
            }
        ),
        encoding="utf-8",
    )
    loaded = DatasetGradingManifest.from_path(manifest)
    assert loaded.config_for("finance").descriptions[0] == "F-1"
    assert loaded.config_for("shougang").descriptions[0] == "S-1"
    assert loaded.sha256_for("finance") != loaded.sha256_for("shougang")


def test_manifest_rejects_non_data_level_ground_truth_field(tmp_path: Path) -> None:
    finance = tmp_path / "finance.json"
    shougang = tmp_path / "shougang.json"
    finance.write_text(
        json.dumps({"levels": ["L1"], "descriptions": ["F"], "gt_field": "other"}),
        encoding="utf-8",
    )
    shougang.write_text(
        json.dumps({"levels": ["L1"], "descriptions": ["S"], "gt_field": "data_level"}),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "datasets": {
                    "finance": {"path": finance.name, "sha256": hashlib.sha256(finance.read_bytes()).hexdigest()},
                    "shougang": {"path": shougang.name, "sha256": hashlib.sha256(shougang.read_bytes()).hexdigest()},
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="gt_field data_level"):
        DatasetGradingManifest.from_path(manifest)


def test_manifest_direct_constructor_rejects_non_data_level_configs(tmp_path: Path) -> None:
    config = GradingConfig(
        levels=("L1",), descriptions=("synthetic",), gt_field="other"
    )
    with pytest.raises(ValueError, match="gt_field data_level"):
        DatasetGradingManifest(
            configs={"finance": config, "shougang": config},
            hashes={"finance": "a" * 64, "shougang": "b" * 64},
            paths={"finance": tmp_path / "finance.json", "shougang": tmp_path / "shougang.json"},
            source_path=tmp_path / "manifest.json",
        )


def test_manifest_rejects_missing_extra_or_tampered_standards(tmp_path: Path) -> None:
    finance = tmp_path / "finance.json"
    digest = _write_grading(finance, "F")
    missing = tmp_path / "missing.json"
    missing.write_text(
        json.dumps({"datasets": {"finance": {"path": finance.name, "sha256": digest}}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="exactly finance and shougang"):
        DatasetGradingManifest.from_path(missing)

    shougang = tmp_path / "shougang.json"
    shougang_digest = _write_grading(shougang, "S")
    valid = tmp_path / "valid.json"
    valid.write_text(
        json.dumps(
            {
                "datasets": {
                    "finance": {"path": finance.name, "sha256": digest},
                    "shougang": {"path": shougang.name, "sha256": shougang_digest},
                }
            }
        ),
        encoding="utf-8",
    )
    shougang.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="sha256 mismatch"):
        DatasetGradingManifest.from_path(valid)
