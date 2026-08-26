"""Publish a validated per-dataset RL source release atomically.

The five-field VeRL rows are first written to a hidden sibling staging
folder, validated with the same registry/corpus/task/grading contract, hashed,
and only then renamed to ``--output-dir``. Existing outputs are never reused.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

from agent.hashing import sha256_file
from agent.task import GradingConfig, LeafRegistry, TaskConfig
from agent.task.assets import load_corpus_categories
from agent.training.rl import export_rl_dataset, validate_rl_dataset


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", required=True)
    parser.add_argument("--split-dir")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--metadata-fields", nargs="+", required=True)
    parser.add_argument("--grading-config")
    parser.add_argument("--allow-label-gaps", nargs="*", default=())
    parser.add_argument("--allow-any-label-gap", action="store_true")
    parser.add_argument("--task-config")
    parser.add_argument(
        "--failed-audit", default=None,
        help="Separate failure JSON; defaults to <output-dir>.failed.json",
    )
    return parser.parse_args(argv)


def _task_config(path: str | None, fields: list[str]) -> TaskConfig:
    value: dict[str, Any] = {}
    if path:
        loaded = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("task config must be a JSON object")
        value = loaded
    value["metadata_fields"] = fields
    return TaskConfig.from_mapping(value)


def _audit_path(output: Path, requested: str | None) -> Path:
    base = Path(requested) if requested else output.with_name(f"{output.name}.failed.json")
    base.parent.mkdir(parents=True, exist_ok=True)
    if not base.exists():
        return base
    for index in range(1, 10_000):
        candidate = base.with_name(f"{base.stem}.{index}{base.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError("unable to allocate unique failed audit path")


def _write_failure(
    output: Path,
    requested: str | None,
    *,
    phase: str,
    error: Exception,
    validation: dict[str, Any] | None,
) -> None:
    try:
        path = _audit_path(output, requested)
        path.write_text(
            json.dumps(
                {
                    "status": "failed",
                    "phase": phase,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "validation": validation,
                    "recorded_at": datetime.now(timezone.utc).isoformat(),
                },
                ensure_ascii=False,
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )
    except OSError as audit_error:
        print(f"failed to write RL audit: {type(audit_error).__name__}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output = Path(args.output_dir)
    staging: Path | None = None
    validation: dict[str, Any] | None = None
    phase = "prepare"
    try:
        if output.exists():
            raise FileExistsError("RL release output must be a new path")
        output.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent)
        )
        registry = LeafRegistry.from_path(args.registry)
        config = _task_config(args.task_config, args.metadata_fields)
        corpus = {
            category.category_id: category
            for category in load_corpus_categories(args.corpus)
        }
        grading = (
            GradingConfig.from_path(args.grading_config)
            if args.grading_config else None
        )
        grading_sha256 = sha256_file(args.grading_config) if args.grading_config else None
        phase = "export"
        report = export_rl_dataset(
            args.canonical,
            args.split_dir,
            staging,
            args.dataset,
            registry,
            config,
            corpus=corpus,
            grading=grading,
            allow_label_gaps=args.allow_label_gaps,
            allow_any_label_gap=args.allow_any_label_gap,
        )
        if grading is not None:
            grading_report = report.get("grading")
            if isinstance(grading_report, dict):
                grading_report["standard_sha256"] = grading_sha256
        phase = "validate"
        validation = validate_rl_dataset(
            staging,
            args.dataset,
            registry,
            config,
            corpus=corpus,
            grading=grading,
        )
        if not validation.get("valid", False):
            raise ValueError("RL release validation failed")
        phase = "report"
        for split in ("train", "val", "test"):
            parquet = staging / f"{split}.parquet"
            if not parquet.is_file():
                raise ValueError(f"RL export did not produce {split}.parquet")
            details = report["splits"][split]
            details["output_file"] = (output / parquet.name).as_posix()
            details["parquet_sha256"] = sha256_file(parquet)
        report["validation"] = validation
        report["release"] = {"status": "passed", "published": True}
        (staging / "export_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        phase = "publish"
        os.replace(staging, output)
        staging = None
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        _write_failure(
            output,
            args.failed_audit,
            phase=phase,
            error=exc,
            validation=validation,
        )
        print(f"export_rl_dataset: {exc}", file=sys.stderr)
        return 2
    finally:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
