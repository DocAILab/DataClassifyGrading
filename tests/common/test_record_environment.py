"""Deterministic, secret-free training environment manifest."""

from __future__ import annotations

import json
from pathlib import Path

from script.verl.common.record_environment import collect_environment, main


def test_collect_environment_records_fixed_packages_and_gpu_without_env_dump() -> None:
    versions = {
        name: f"version-{name}"
        for name in ("verl", "vllm", "ray", "torch", "transformers", "peft")
    }
    manifest = collect_environment(
        gpu_target="RTX PRO 6000 96GB",
        package_version=lambda name: versions[name],
        torch_info=lambda: {
            "imported": True,
            "cuda_available": True,
            "cuda_runtime": "12.8",
            "device_count": 1,
            "device_name": "Synthetic GPU",
            "total_vram_bytes": 96 * 1024**3,
        },
    )
    assert manifest["format"] == "dataclassify-training-environment-v1"
    assert manifest["gpu_target"] == "RTX PRO 6000 96GB"
    assert manifest["packages"]["verl"] == "version-verl"
    assert manifest["torch_cuda"]["total_vram_bytes"] == 96 * 1024**3
    rendered = json.dumps(manifest).lower()
    for forbidden in ("environment_variables", "token", "password", "secret"):
        assert forbidden not in rendered


def test_cli_writes_atomic_json_and_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "environment.json"
    assert main(["--output", str(output), "--gpu-target", "CPU test host"]) == 0
    stored = json.loads(output.read_text(encoding="utf-8"))
    assert stored["gpu_target"] == "CPU test host"
    assert set(stored["packages"]) == {
        "peft", "ray", "torch", "transformers", "verl", "vllm"
    }
    assert main(["--output", str(output), "--gpu-target", "CPU test host"]) == 2
