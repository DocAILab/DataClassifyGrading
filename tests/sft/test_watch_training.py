from __future__ import annotations

import json
from pathlib import Path

import pytest

from script.verl.sft.watch_training import (
    inspect_completed_training,
    write_status_atomic,
)


def _checkpoint(root: Path, step: int = 2) -> Path:
    checkpoint = root / f"global_step_{step}"
    checkpoint.mkdir(parents=True)
    (checkpoint / "model_world_size_1_rank_0.pt").write_bytes(b"adapter-state")
    return checkpoint


def test_certifies_finite_loss_and_materializes_merge_metadata(tmp_path: Path) -> None:
    log = tmp_path / "train.log"
    log.write_text("step=1 'train/loss': 2.5\nstep=2 train/loss=1.25\n", encoding="utf-8")
    checkpoint = _checkpoint(tmp_path / "checkpoints")

    result = inspect_completed_training(
        log_path=log,
        checkpoint_root=checkpoint.parent,
        minimum_step=2,
        lora_rank=8,
        lora_alpha=16,
    )

    assert result["status"] == "completed"
    assert result["latest_step"] == 2
    assert result["losses"] == [2.5, 1.25]
    assert len(result["model_sha256"]) == 64
    assert json.loads((checkpoint / "lora_train_meta.json").read_text()) == {
        "r": 8,
        "lora_alpha": 16,
    }


@pytest.mark.parametrize("text", ["CUDA out of memory", "Traceback (most recent call last):", "train/loss=nan"])
def test_rejects_failure_markers(tmp_path: Path, text: str) -> None:
    log = tmp_path / "train.log"
    log.write_text(text, encoding="utf-8")
    _checkpoint(tmp_path / "checkpoints")

    with pytest.raises(ValueError, match="failure marker|finite loss"):
        inspect_completed_training(
            log_path=log,
            checkpoint_root=tmp_path / "checkpoints",
            minimum_step=1,
            lora_rank=8,
            lora_alpha=16,
        )


def test_rejects_exit_without_checkpoint(tmp_path: Path) -> None:
    log = tmp_path / "train.log"
    log.write_text("train/loss=1.0\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="checkpoint"):
        inspect_completed_training(
            log_path=log,
            checkpoint_root=tmp_path / "checkpoints",
            minimum_step=1,
            lora_rank=8,
            lora_alpha=16,
        )


def test_status_write_is_atomic_and_replaces_previous_value(tmp_path: Path) -> None:
    path = tmp_path / "status.json"
    write_status_atomic(path, {"status": "running"})
    write_status_atomic(path, {"status": "completed", "step": 2})

    assert json.loads(path.read_text()) == {"status": "completed", "step": 2}
    assert not path.with_name(".status.json.tmp").exists()
