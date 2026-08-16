from pathlib import Path
import os
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "script" / "verl" / "run_sft.sh"


def _run_launcher(*overrides: str) -> subprocess.CompletedProcess[str]:
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash is required for launcher tests")
    env = os.environ.copy()
    env["PYTHON_BIN"] = shutil.which("true") or "true"
    return subprocess.run(
        [bash, str(LAUNCHER), *overrides],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_sdpa_requires_remove_padding_to_be_disabled() -> None:
    result = _run_launcher("+model.override_config.attn_implementation=sdpa")

    assert result.returncode == 2
    assert "model.use_remove_padding=false" in result.stderr


def test_sdpa_accepts_explicit_safe_remove_padding_override() -> None:
    result = _run_launcher(
        "+model.override_config.attn_implementation=sdpa",
        "model.use_remove_padding=false",
    )

    assert result.returncode == 0
