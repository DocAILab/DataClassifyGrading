"""Record the selected training Python/CUDA environment without secrets.

Run this module with the exact server interpreter used by VeRL. The manifest
contains no environment-variable dump, credentials, timestamps, or machine
paths beyond the selected Python executable.
"""

from __future__ import annotations

import argparse
from importlib import metadata
import json
import platform
from pathlib import Path
import sys
from typing import Any, Callable

_PACKAGE_NAMES = ("verl", "vllm", "ray", "torch", "transformers", "peft")


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _torch_info() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:  # environment probe must report rather than crash
        return {
            "imported": False,
            "error_type": type(exc).__name__,
            "cuda_available": False,
            "cuda_runtime": None,
            "device_count": 0,
            "device_name": None,
            "total_vram_bytes": None,
        }
    available = bool(torch.cuda.is_available())
    count = int(torch.cuda.device_count()) if available else 0
    name = None
    total_vram = None
    if count:
        properties = torch.cuda.get_device_properties(0)
        name = str(properties.name)
        total_vram = int(properties.total_memory)
    return {
        "imported": True,
        "cuda_available": available,
        "cuda_runtime": getattr(torch.version, "cuda", None),
        "device_count": count,
        "device_name": name,
        "total_vram_bytes": total_vram,
    }


def collect_environment(
    *,
    gpu_target: str,
    package_version: Callable[[str], str | None] = _package_version,
    torch_info: Callable[[], dict[str, Any]] = _torch_info,
) -> dict[str, Any]:
    """Return a deterministic manifest of explicit runtime facts."""

    if not isinstance(gpu_target, str) or not gpu_target.strip():
        raise ValueError("gpu_target must be a non-empty string")
    return {
        "format": "dataclassify-training-environment-v1",
        "gpu_target": gpu_target.strip(),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": str(Path(sys.executable).resolve()),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "packages": {
            name: package_version(name) for name in sorted(_PACKAGE_NAMES)
        },
        "torch_cuda": torch_info(),
    }


def _args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpu-target", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _args(argv)
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        print("status:error output_exists", file=sys.stderr)
        return 2
    try:
        manifest = collect_environment(gpu_target=args.gpu_target)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.tmp")
        try:
            temporary.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            temporary.replace(output)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    except (OSError, ValueError) as exc:
        print(f"status:error type={type(exc).__name__}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": "ok",
                "cuda_available": manifest["torch_cuda"]["cuda_available"],
                "device_count": manifest["torch_cuda"]["device_count"],
            },
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
