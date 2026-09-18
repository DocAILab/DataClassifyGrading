"""Periodically monitor a background VeRL SFT run and certify its checkpoint."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Any

from agent.hashing import sha256_file


_LOSS_RE = re.compile(
    r"(?:train/)?loss['\"]?\s*[:=]\s*"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)
_NONFINITE_LOSS_RE = re.compile(
    r"(?:train/)?loss['\"]?\s*[:=]\s*(?:nan|[-+]?inf(?:inity)?)",
    re.IGNORECASE,
)
_FAILURE_MARKERS = (
    "traceback (most recent call last):",
    "cuda out of memory",
    "torch.cuda.outofmemoryerror",
    "processfailedexception",
    "childfailederror",
)
_STEP_RE = re.compile(r"global_step_(\d+)$")
_MODEL_RE = re.compile(r"model_world_size_(\d+)_rank_(\d+)\.pt$")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_status_atomic(path: str | Path, value: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, destination)


def _latest_checkpoint(root: Path, minimum_step: int) -> tuple[Path, int]:
    if not root.is_dir():
        raise FileNotFoundError(f"checkpoint directory not found: {root}")
    candidates: list[tuple[int, Path]] = []
    for path in root.iterdir():
        match = _STEP_RE.fullmatch(path.name)
        if path.is_dir() and match:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        raise FileNotFoundError(f"no global_step checkpoint found under {root}")
    step, checkpoint = max(candidates)
    if step < minimum_step:
        raise ValueError(
            f"latest checkpoint step {step} is below required minimum {minimum_step}"
        )
    return checkpoint, step


def _single_rank_model(checkpoint: Path) -> Path:
    files = [
        path
        for path in checkpoint.iterdir()
        if path.is_file() and _MODEL_RE.fullmatch(path.name)
    ]
    if len(files) != 1:
        raise ValueError(
            "checkpoint must contain exactly one model_world_size_*_rank_*.pt file"
        )
    match = _MODEL_RE.fullmatch(files[0].name)
    assert match is not None
    if (int(match.group(1)), int(match.group(2))) != (1, 0):
        raise ValueError("checkpoint must use world_size=1 and rank=0")
    if files[0].stat().st_size <= 0:
        raise ValueError("checkpoint model file is empty")
    return files[0]


def inspect_completed_training(
    *,
    log_path: str | Path,
    checkpoint_root: str | Path,
    minimum_step: int,
    lora_rank: int,
    lora_alpha: int,
) -> dict[str, Any]:
    log = Path(log_path)
    if not log.is_file():
        raise FileNotFoundError(f"training log not found: {log}")
    text = log.read_text(encoding="utf-8", errors="replace")
    lowered = text.lower()
    marker = next((item for item in _FAILURE_MARKERS if item in lowered), None)
    if marker or _NONFINITE_LOSS_RE.search(text):
        raise ValueError(f"training log contains failure marker: {marker or 'non-finite loss'}")
    losses = [float(match.group(1)) for match in _LOSS_RE.finditer(text)]
    if not losses or any(not math.isfinite(value) for value in losses):
        raise ValueError("training log does not contain a finite loss")

    checkpoint, step = _latest_checkpoint(Path(checkpoint_root), minimum_step)
    model_file = _single_rank_model(checkpoint)
    meta = {"r": lora_rank, "lora_alpha": lora_alpha}
    write_status_atomic(checkpoint / "lora_train_meta.json", meta)
    return {
        "status": "completed",
        "recorded_at": _utc_now(),
        "latest_step": step,
        "checkpoint": str(checkpoint),
        "model_file": str(model_file),
        "model_bytes": model_file.stat().st_size,
        "model_sha256": sha256_file(model_file),
        "log": str(log),
        "log_sha256": sha256_file(log),
        "losses": losses,
        "last_loss": losses[-1],
        "lora": meta,
    }


def _pid_alive(pid: int) -> bool:
    stat = Path(f"/proc/{pid}/stat")
    if stat.is_file():
        try:
            if stat.read_text(encoding="utf-8").split()[2] == "Z":
                return False
        except (OSError, IndexError):
            pass
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _tail(path: Path, lines: int = 40) -> list[str]:
    if not path.is_file():
        return []
    return path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]


def watch(
    *,
    pid: int,
    log_path: Path,
    checkpoint_root: Path,
    status_path: Path,
    minimum_step: int,
    lora_rank: int,
    lora_alpha: int,
    interval_seconds: float,
) -> int:
    while _pid_alive(pid):
        write_status_atomic(
            status_path,
            {
                "status": "running",
                "recorded_at": _utc_now(),
                "pid": pid,
                "log": str(log_path),
                "checkpoint_root": str(checkpoint_root),
                "log_tail": _tail(log_path),
            },
        )
        time.sleep(interval_seconds)
    try:
        completed = inspect_completed_training(
            log_path=log_path,
            checkpoint_root=checkpoint_root,
            minimum_step=minimum_step,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
        )
    except (OSError, ValueError) as exc:
        write_status_atomic(
            status_path,
            {
                "status": "failed",
                "recorded_at": _utc_now(),
                "pid": pid,
                "reason": str(exc),
                "log_tail": _tail(log_path),
            },
        )
        return 2
    write_status_atomic(status_path, completed)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--minimum-step", type=int, default=1)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--interval-seconds", type=float, default=60.0)
    args = parser.parse_args(argv)
    for name in ("pid", "minimum_step", "lora_rank", "lora_alpha"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.interval_seconds <= 0:
        parser.error("--interval-seconds must be positive")
    return watch(
        pid=args.pid,
        log_path=Path(args.log),
        checkpoint_root=Path(args.checkpoint_root),
        status_path=Path(args.status),
        minimum_step=args.minimum_step,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        interval_seconds=args.interval_seconds,
    )


if __name__ == "__main__":
    raise SystemExit(main())
