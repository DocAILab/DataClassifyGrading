"""Fail-closed verification for a real one-update DPO smoke run."""

from __future__ import annotations

import json
import math
from pathlib import Path


def _finite_positive(value, name: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and positive") from exc
    if not math.isfinite(numeric) or numeric <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return numeric


def verify_smoke(training_dir: str | Path, verification_dir: str | Path) -> dict:
    """Create SMOKE_VERIFIED only after all update and artifact invariants pass."""
    training = Path(training_dir)
    report_path = training / "training_report.json"
    metrics_path = training / "trainer_metrics.jsonl"
    if not report_path.is_file() or not metrics_path.is_file():
        raise ValueError("smoke requires training report and trainer metrics")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("completed") is not True:
        raise ValueError("smoke training did not complete")
    steps = int(report.get("optimizer_steps", 0))
    if steps < 1:
        raise ValueError("smoke requires at least one optimizer step")
    train_loss = report.get("train_metrics", {}).get("train_loss")
    if train_loss is None or not math.isfinite(float(train_loss)):
        raise ValueError("smoke loss must be finite")
    policy_update = _finite_positive(
        report.get("policy_update", {}).get("absolute_update_norm"),
        "policy update norm",
    )
    if report.get("reference_unchanged") is not True:
        raise ValueError("reference parameters changed during smoke")
    finite_grad_norms: list[float] = []
    with metrics_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            value = row.get("grad_norm")
            if value is not None:
                numeric = float(value)
                if math.isfinite(numeric) and numeric > 0:
                    finite_grad_norms.append(numeric)
    if not finite_grad_norms:
        raise ValueError("smoke requires a finite positive grad_norm metric")
    adapter = training / "final_adapter"
    required_adapter = [adapter / "adapter_config.json", adapter / "adapter_model.safetensors"]
    if any(not path.is_file() or path.stat().st_size == 0 for path in required_adapter):
        raise ValueError("smoke adapter is missing or empty")
    checkpoints = sorted(training.glob("checkpoint-*"))
    if not checkpoints or not any((path / "trainer_state.json").is_file() for path in checkpoints):
        raise ValueError("smoke checkpoint with trainer_state.json is missing")
    verified = {
        "valid": True,
        "optimizer_steps": steps,
        "train_loss": float(train_loss),
        "max_grad_norm_observed": max(finite_grad_norms),
        "policy_absolute_update_norm": policy_update,
        "reference_unchanged": True,
        "adapter": str(adapter.resolve()),
        "checkpoints": [str(path.resolve()) for path in checkpoints],
    }
    output = Path(verification_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "smoke_report.json").write_text(
        json.dumps(verified, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "SMOKE_VERIFIED").write_text("verified\n", encoding="utf-8")
    return verified
