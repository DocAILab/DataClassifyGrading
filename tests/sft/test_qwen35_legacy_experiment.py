from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from script.verl.sft.qwen35_legacy_experiment import (
    ExperimentConfig,
    _probe_python,
    build_command,
    validate_model_snapshot,
    validate_sft_parquet,
)


def test_python_probe_imports_actual_sft_entrypoint(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: dict[str, object] = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(
                {
                    "verl": "0.9.0",
                    "torch": "2.8.0",
                    "cuda": "12.8",
                    "cuda_available": True,
                    "site_packages": "/venv/site-packages",
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    _probe_python(Path("/venv/bin/python"))

    command = observed["command"]
    assert isinstance(command, list)
    assert "import verl.trainer.sft_trainer" in command[-1]


def _config(tmp_path: Path, *, mode: str = "formal") -> ExperimentConfig:
    repo = tmp_path / "repo"
    (repo / "script/verl/sft").mkdir(parents=True)
    (repo / "script/verl/sft/run.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    return ExperimentConfig(
        repo_root=repo,
        python_bin=Path("/venv/bin/python"),
        model_path=tmp_path / "model",
        train_file=tmp_path / "train.parquet",
        val_file=tmp_path / "val.parquet",
        output_dir=tmp_path / "output",
        mode=mode,
    )


def test_formal_command_locks_approved_training_contract(tmp_path: Path) -> None:
    config = _config(tmp_path)

    command = build_command(config)
    joined = "\n".join(command)

    assert command[:4] == [
        "bash",
        str(config.repo_root / "script/verl/sft/run.sh"),
        f"data.train_files={config.train_file}",
        f"data.val_files={config.val_file}",
    ]
    for expected in (
        "data.train_batch_size=32",
        "data.micro_batch_size_per_gpu=1",
        "data.max_length=4096",
        "data.max_token_len_per_gpu=4096",
        "data.truncation=error",
        f"model.path={config.model_path}",
        "model.enable_gradient_checkpointing=true",
        "model.lora_rank=8",
        "model.lora_alpha=16",
        "model.target_modules=all-linear",
        "engine.model_dtype=bfloat16",
        "optim.lr=0.0001",
        "trainer.total_epochs=1",
        "trainer.save_freq=-1",
        "trainer.test_freq=-1",
        "trainer.resume_mode=disable",
    ):
        assert expected in command, joined
    assert not any("total_training_steps=" in item for item in command)


def test_smoke_command_is_exactly_two_steps(tmp_path: Path) -> None:
    command = build_command(_config(tmp_path, mode="smoke"))

    assert "trainer.total_training_steps=2" in command
    assert "data.train_batch_size=2" in command
    assert "model.enable_gradient_checkpointing=false" in command


def test_public_config_has_no_environment_or_credentials(tmp_path: Path) -> None:
    config = _config(tmp_path)
    rendered = json.dumps(config.to_mapping(), sort_keys=True)

    assert "password" not in rendered.lower()
    assert "token" not in rendered.lower()
    assert "environment" not in rendered.lower()
    assert config.to_mapping()["seed"] == 42


def test_validate_model_snapshot_requires_every_indexed_shard(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    (model / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "one.safetensors", "b": "two.safetensors"}}),
        encoding="utf-8",
    )
    (model / "one.safetensors").write_bytes(b"one")

    with pytest.raises(FileNotFoundError, match="two.safetensors"):
        validate_model_snapshot(model)

    (model / "two.safetensors").write_bytes(b"two")
    result = validate_model_snapshot(model)
    assert result["shard_count"] == 2
    assert result["total_shard_bytes"] == 6


def _write_parquet(path: Path, rows: int) -> None:
    messages = [
        {"role": "system", "content": "contract"},
        {"role": "user", "content": "metadata"},
        {"role": "assistant", "content": "{}"},
    ]
    table = pa.table(
        {
            "messages": [messages] * rows,
            "stage": ["stage1", "stage2"] * (rows // 2),
            "source_id": [f"s-{index // 2}" for index in range(rows)],
        }
    )
    pq.write_table(table, path)


def test_validate_sft_parquet_checks_rows_and_stage_pairs(tmp_path: Path) -> None:
    path = tmp_path / "train.parquet"
    _write_parquet(path, 6)

    report = validate_sft_parquet(path, expected_rows=6)

    assert report["rows"] == 6
    assert report["sources"] == 3
    assert report["stage_counts"] == {"stage1": 3, "stage2": 3}


def test_validate_sft_parquet_rejects_unexpected_formal_count(tmp_path: Path) -> None:
    path = tmp_path / "train.parquet"
    _write_parquet(path, 4)

    with pytest.raises(ValueError, match="expected 8664"):
        validate_sft_parquet(path, expected_rows=8664)


def test_new_output_directory_must_be_empty(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.output_dir.mkdir()
    (config.output_dir / "existing.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="non-empty"):
        config.validate_output_policy()
