"""Watch a background true-E2E evaluation and certify its final report."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from typing import Any

from agent.hashing import sha256_file


def _count(rows: list[dict[str, Any]], field: str) -> int:
    return sum(bool(row.get(field)) for row in rows)


def validate_report(report_path: str | Path, *, expected_sources: int) -> dict[str, Any]:
    """Load a completed report and independently verify its core counters."""
    path = Path(report_path)
    with path.open(encoding="utf-8") as handle:
        report = json.load(handle)
    if not isinstance(report, dict):
        raise ValueError("evaluation report must be a JSON object")
    metrics = report.get("metrics")
    rows = report.get("per_source")
    if not isinstance(metrics, dict) or not isinstance(rows, list):
        raise ValueError("evaluation report requires metrics and per_source")
    if len(rows) != expected_sources:
        raise ValueError(
            f"expected {expected_sources} per_source rows, found {len(rows)}"
        )
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("every per_source item must be an object")
    source_ids = [str(row.get("source_id", "")) for row in rows]
    if any(not source_id for source_id in source_ids):
        raise ValueError("every per_source item requires source_id")
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("duplicate source_id values in per_source")

    recomputed = {
        "stage1_format_valid_count": _count(rows, "stage1_format_valid"),
        "stage1_contract_valid_count": _count(rows, "stage1_contract_valid"),
        "stage1_recalled_count": _count(rows, "recalled"),
        "stage2_attempted_count": _count(rows, "stage2_attempted"),
        "stage2_format_valid_count": _count(rows, "stage2_format_valid"),
        "stage2_contract_valid_count": _count(rows, "stage2_contract_valid"),
        "leaf_correct_count": _count(rows, "leaf_correct"),
        "level_correct_count": _count(rows, "level_correct"),
        "joint_correct_count": _count(rows, "e2e_correct"),
        "failure_count": sum(len(row.get("failures") or ()) for row in rows),
    }
    comparisons = {
        "sources": expected_sources,
        "stage1_recalled_count": recomputed["stage1_recalled_count"],
        "stage2_attempted": recomputed["stage2_attempted_count"],
        "strict_joint_em_count": recomputed["joint_correct_count"],
        "true_e2e_correct": recomputed["joint_correct_count"],
    }
    for field, expected in comparisons.items():
        if metrics.get(field) != expected:
            raise ValueError(
                f"reported {field}={metrics.get(field)!r} does not match "
                f"recomputed value {expected}"
            )

    return {
        "status": "completed",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "report": str(path),
        "report_sha256": sha256_file(path),
        "expected_sources": expected_sources,
        "observed_sources": len(rows),
        "unique_source_ids": len(set(source_ids)),
        "recomputed": recomputed,
        "metrics": metrics,
    }


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _write_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def watch(
    *,
    pid: int,
    report: str | Path,
    status_file: str | Path,
    expected_sources: int,
    interval_seconds: float,
) -> int:
    """Poll until a valid report appears or the producer exits unsuccessfully."""
    report_path = Path(report)
    destination = Path(status_file)
    while True:
        if report_path.is_file():
            try:
                status = validate_report(
                    report_path, expected_sources=expected_sources
                )
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                if _pid_alive(pid):
                    time.sleep(interval_seconds)
                    continue
                status = {
                    "status": "failed",
                    "recorded_at": datetime.now(timezone.utc).isoformat(),
                    "pid": pid,
                    "report": str(report_path),
                    "reason": f"invalid final report: {exc}",
                }
                _write_atomic(destination, status)
                return 2
            _write_atomic(destination, status)
            return 0
        if not _pid_alive(pid):
            status = {
                "status": "failed",
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                "pid": pid,
                "report": str(report_path),
                "reason": "evaluation process exited before producing a report",
            }
            _write_atomic(destination, status)
            return 2
        time.sleep(interval_seconds)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--status-file", required=True)
    parser.add_argument("--expected-sources", type=int, required=True)
    parser.add_argument("--interval-seconds", type=float, default=60.0)
    args = parser.parse_args(argv)
    if args.pid <= 0:
        parser.error("--pid must be positive")
    if args.expected_sources <= 0:
        parser.error("--expected-sources must be positive")
    if args.interval_seconds <= 0:
        parser.error("--interval-seconds must be positive")
    return watch(
        pid=args.pid,
        report=args.report,
        status_file=args.status_file,
        expected_sources=args.expected_sources,
        interval_seconds=args.interval_seconds,
    )


if __name__ == "__main__":
    raise SystemExit(main())
