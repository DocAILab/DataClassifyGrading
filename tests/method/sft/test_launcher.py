from pathlib import Path
import json
import os
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[3]
LAUNCHER = ROOT / "src" / "method" / "sft" / "script" / "run.sh"


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


def test_official_smoke_defaults_to_field_name_only() -> None:
    task = json.loads(
        (ROOT / "tests" / "method" / "sft" / "fixtures" / "task.json").read_text(
            encoding="utf-8"
        )
    )
    launcher = (ROOT / "src" / "method" / "sft" / "script" / "smoke.sh").read_text(
        encoding="utf-8"
    )

    assert task["metadata_fields"] == ["field_name"]
    assert "--metadata-fields field_name field_description" not in launcher
    assert launcher.count("--metadata-fields field_name") == 2
