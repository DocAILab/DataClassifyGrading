"""Fail-closed checks around the verified single-rank LoRA merge seam."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from script.verl.sft.merge_lora_checkpoint import (
    discover_single_rank_checkpoint,
    lora_spec,
    require_new_output_dir,
)


def _checkpoint(root: Path, *, world_size: int = 1) -> Path:
    root.mkdir()
    (root / f"model_world_size_{world_size}_rank_0.pt").write_bytes(b"state")
    (root / "lora_train_meta.json").write_text(
        json.dumps({"r": 8, "lora_alpha": 16}), encoding="utf-8"
    )
    return root


def test_discovers_only_verified_single_rank_layout(tmp_path: Path) -> None:
    model_file, meta = discover_single_rank_checkpoint(_checkpoint(tmp_path / "ok"))
    assert model_file.name == "model_world_size_1_rank_0.pt"
    assert meta == {"r": 8, "lora_alpha": 16}

    with pytest.raises(ValueError, match="world_size=1"):
        discover_single_rank_checkpoint(_checkpoint(tmp_path / "multi", world_size=2))


def test_checkpoint_requires_meta_and_unambiguous_model_file(tmp_path: Path) -> None:
    missing_meta = tmp_path / "missing-meta"
    missing_meta.mkdir()
    (missing_meta / "model_world_size_1_rank_0.pt").write_bytes(b"state")
    with pytest.raises(FileNotFoundError, match="lora_train_meta"):
        discover_single_rank_checkpoint(missing_meta)

    ambiguous = _checkpoint(tmp_path / "ambiguous")
    (ambiguous / "model_world_size_1_rank_1.pt").write_bytes(b"other")
    with pytest.raises(ValueError, match="exactly one"):
        discover_single_rank_checkpoint(ambiguous)


def test_lora_spec_requires_valid_meta_and_lora_keys() -> None:
    keys = [
        "base_model.model.layers.0.q_proj.lora_A.default.weight",
        "base_model.model.layers.0.q_proj.lora_B.default.weight",
    ]
    assert lora_spec(keys, {"r": 8, "lora_alpha": 16}) == (["q_proj"], 8, 16)

    with pytest.raises(ValueError, match="no LoRA keys"):
        lora_spec(["model.embed.weight"], {"r": 8, "lora_alpha": 16})
    with pytest.raises(ValueError, match="positive integer"):
        lora_spec(keys, {"r": 0, "lora_alpha": 16})


def test_merge_output_must_be_new(tmp_path: Path) -> None:
    output = tmp_path / "merged"
    require_new_output_dir(output)
    output.mkdir()
    with pytest.raises(FileExistsError, match="must not already exist"):
        require_new_output_dir(output)
