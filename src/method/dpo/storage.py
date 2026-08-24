"""Storage headroom and explicit cleanup audit helpers."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Mapping, Sequence


def free_bytes(path: str | Path) -> int:
    return int(shutil.disk_usage(Path(path)).free)


def assert_training_capacity(
    persistent_path: str | Path,
    shm_path: str | Path = "/dev/shm",
    *,
    minimum_persistent_free_bytes: int = 20 * 2**30,
    minimum_shm_free_bytes: int = 35 * 2**30,
) -> dict:
    persistent_free = free_bytes(persistent_path)
    shm_free = free_bytes(shm_path)
    if persistent_free < minimum_persistent_free_bytes:
        raise RuntimeError(
            f"persistent storage has {persistent_free} bytes free; "
            f"requires {minimum_persistent_free_bytes}"
        )
    if shm_free < minimum_shm_free_bytes:
        raise RuntimeError(
            f"shared memory has {shm_free} bytes free; requires {minimum_shm_free_bytes}"
        )
    return {
        "valid": True,
        "persistent_path": str(Path(persistent_path).resolve()),
        "persistent_free_bytes": persistent_free,
        "minimum_persistent_free_bytes": minimum_persistent_free_bytes,
        "shm_path": str(Path(shm_path).resolve()),
        "shm_free_bytes": shm_free,
        "minimum_shm_free_bytes": minimum_shm_free_bytes,
    }


def write_cleanup_report(
    output_path: str | Path, deleted: Sequence[Mapping[str, object]]
) -> dict:
    rows = [dict(item) for item in deleted]
    for item in rows:
        path_value = item.get("path")
        is_explicit_absolute = isinstance(path_value, str) and (
            Path(path_value).is_absolute() or path_value.startswith("/")
        )
        if not is_explicit_absolute:
            raise ValueError("cleanup report paths must be explicit absolute paths")
        if int(item.get("bytes", -1)) < 0:
            raise ValueError("cleanup report bytes must be non-negative")
        if not str(item.get("recoverable_from", "")).strip():
            raise ValueError("cleanup report requires recovery provenance")
    report = {
        "deleted": rows,
        "reclaimed_bytes": sum(int(item["bytes"]) for item in rows),
        "datasets_deleted": False,
        "base_models_deleted": False,
        "permanent_adapters_deleted": False,
        "reports_deleted": False,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report
