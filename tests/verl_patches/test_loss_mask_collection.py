"""Local collection-order guards for ``test_loss_mask_behavior.py``.

These tests never import torch or VeRL.  They re-run pytest on the behavior
module in a subprocess with a poisoned ``sys.path``:

- missing ``verl`` metadata -> the module must skip at collection time *before*
  torch/verl are imported (a ``torch.py`` shim that raises on import must not
  be touched);
- wrong ``verl`` version -> the module must fail fast at collection (a
  9.9.9 dist-info must produce a collection error, not a silent skip).

They are skipped when a real ``verl==0.9.0`` runtime is present (server venv),
because there the poison shims cannot shadow the real import graph.
"""

from __future__ import annotations

import importlib.metadata
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BEHAVIOR_MODULE = "tests/verl_patches/test_loss_mask_behavior.py"

try:
    VERL_VERSION = importlib.metadata.version("verl")
except importlib.metadata.PackageNotFoundError:
    VERL_VERSION = None

_RUNTIME_PRESENT = (
    "verl runtime present; local-collection guard not exercised "
    "(run on a checkout without verl, or with a broken-verl shim)"
)


def _run_collection_probe(shim: Path | None, marker: Path | None) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["LOSS_MASK_SUBPROCESS"] = "1"
    if shim is not None:
        env["PYTHONPATH"] = os.pathsep.join(
            [str(shim), env.get("PYTHONPATH", "")]
        )
    if marker is not None:
        env["LOSS_MASK_TORCH_MARKER"] = str(marker)
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            BEHAVIOR_MODULE,
            "-q",
            "-p",
            "no:cacheprovider",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_missing_verl_metadata_skips_before_torch_import(tmp_path) -> None:
    """Collection must decide on verl metadata before importing torch/verl.

    The shim ``torch.py`` raises on import and writes a marker if it is ever
    loaded; a skip-before-import guard leaves the marker absent and pytest
    exits 0 with skipped items.
    """

    if VERL_VERSION is not None:
        pytest.skip(_RUNTIME_PRESENT)

    shim = tmp_path / "shim"
    shim.mkdir()
    marker = tmp_path / "torch-imported.marker"
    (shim / "torch.py").write_text(
        "import os, pathlib\n"
        "pathlib.Path(os.environ['LOSS_MASK_TORCH_MARKER']).write_text('imported')\n"
        "raise ImportError('blocked')\n",
        encoding="utf-8",
    )

    proc = _run_collection_probe(shim, marker)

    assert proc.returncode == 0, f"collection must skip cleanly:\n{proc.stdout}\n{proc.stderr}"
    assert "skipped" in proc.stdout
    assert not marker.exists(), "torch was imported during local collection"


def test_wrong_verl_version_fails_fast_at_collection(tmp_path) -> None:
    """A present-but-wrong verl must fail at collection, not skip silently."""

    if VERL_VERSION is not None:
        pytest.skip(_RUNTIME_PRESENT)

    shim = tmp_path / "shim"
    dist = shim / "verl-9.9.9.dist-info"
    dist.mkdir(parents=True)
    (dist / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: verl\nVersion: 9.9.9\n",
        encoding="utf-8",
    )

    proc = _run_collection_probe(shim, None)

    assert proc.returncode != 0, "wrong verl version must fail collection"
    assert "expected patched verl==0.9.0" in proc.stdout + proc.stderr
