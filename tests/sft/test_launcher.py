"""The SFT launcher forwards an explicit VeRL 0.8 Hydra argument array."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[2]


def _bash() -> str:
    if os.name == "nt":
        for program_files in (
            os.environ.get("ProgramFiles"),
            os.environ.get("ProgramFiles(x86)"),
        ):
            if program_files:
                candidate = Path(program_files) / "Git" / "bin" / "bash.exe"
                if candidate.is_file():
                    return str(candidate)
    command = shutil.which("bash")
    if command:
        return command
    pytest.skip("a POSIX bash is required for launcher tests")


def test_launcher_forwards_argv_as_hydra_overrides(tmp_path) -> None:
    bash = _bash()
    command = " ".join(
        [
            "PYTHON_BIN=echo",
            "NUM_GPUS=2",
            "bash script/verl/sft/run.sh",
            "data.train_files=/tmp/train.parquet",
            "model.path=/tmp/model",
            "model.use_remove_padding=false",
            "model.override_config.attn_implementation=sdpa",
            "trainer.logger=[console]",
        ]
    )
    result = subprocess.run(
        [bash, "-c", command],
        text=True,
        capture_output=True,
        check=False,
        cwd=ROOT,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == [
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        "--nproc-per-node=2",
        "-m",
        "verl.trainer.sft_trainer",
        "data.train_files=/tmp/train.parquet",
        "model.path=/tmp/model",
        "model.use_remove_padding=false",
        "model.override_config.attn_implementation=sdpa",
        "trainer.logger=[console]",
    ]


def test_launcher_rejects_sdpa_without_explicit_padding_binding(tmp_path) -> None:
    bash = _bash()
    result = subprocess.run(
        [
            bash,
            "-c",
            "PYTHON_BIN=echo bash script/verl/sft/run.sh "
            "model.override_config.attn_implementation=sdpa",
        ],
        text=True,
        capture_output=True,
        check=False,
        cwd=ROOT,
    )
    assert result.returncode == 2
    assert "model.use_remove_padding=false" in result.stderr
