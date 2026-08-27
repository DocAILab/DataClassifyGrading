"""CLI for exporting canonical records to a published tool-trajectory SFT release.

Mirrors ``script.verl.sft.export``'s verified release boundary: the dataset is
written into a sibling staging directory, validated and hashed, and only then
published under ``--output-dir`` via atomic ``os.replace``. Existing non-empty
release directories are never modified; failed attempts are written to a
separate audit JSON file.

The generated rows are full chat trajectories (system/user/assistant
tool-call/tool result/terminal strict JSON) for the native tool-loop RLOO
baseline. See :mod:`agent.training.sft.tool_trajectories` for the schema,
the four trajectory classes, and the leakage audit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import os
import shutil
import sys
import tempfile
from typing import Any

from agent.hashing import sha256_file
from agent.task import GradingConfig, LeafRegistry, TaskConfig
from agent.task.assets import load_corpus_categories
from agent.training.sft import (
    export_tool_trajectory_dataset,
    validate_tool_trajectory_dataset,
)
from script.verl.sft.export import (
    _is_nonempty,
    _load_task_config,
    _load_corpus,
    _next_audit_path,
    _write_failed_audit,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="shougang",
        help=(
            "Formal release dataset; shougang-only trajectory line (any other "
            "value fails fast)"
        ),
    )
    parser.add_argument(
        "--canonical",
        required=True,
        help="canonical <dataset>/all.json (schema v2 records with embedded splits)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Published destination for split parquet and export_report.json "
            "(required in export mode; ignored in collect mode)"
        ),
    )
    parser.add_argument("--registry", required=True, help="JSON leaf registry")
    parser.add_argument(
        "--corpus",
        required=True,
        help=(
            "Canonical corpus JSON (category_id/name/description/descriptions/"
            "examples); tool results are byte-exact environment outputs over "
            "this corpus"
        ),
    )
    parser.add_argument(
        "--metadata-fields",
        nargs="+",
        required=True,
        help=(
            "Metadata keys visible to the tool-loop prompt (explicit; the "
            "four-vs-two field contract is owner-blocked and never hardcoded)"
        ),
    )
    parser.add_argument(
        "--task-config",
        help=(
            "Optional JSON task config; task_name is retained and CLI "
            "metadata_fields take precedence"
        ),
    )
    parser.add_argument(
        "--grading-config",
        required=True,
        help=(
            "Grading JSON (levels/descriptions/gt_field); REQUIRED because the "
            "terminal assistant JSON carries a sensitivity level"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("export", "collect"),
        default="export",
        help=(
            "export: build the validated parquet release; collect: write "
            "think-free trajectory context JSONL shards for a label-aware "
            "sub-agent to fill think (no parquet is produced)"
        ),
    )
    parser.add_argument(
        "--collect-dir",
        default=None,
        help="Required with --mode collect: output directory for JSONL shards",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=64,
        help="Rows per JSONL shard in collect mode",
    )
    parser.add_argument(
        "--think-source",
        default="mock",
        help=(
            "Think text source: 'mock' (local deterministic, no credentials) "
            "or 'file:<path>' (pre-generated think from collect shards, "
            "filled by the label-aware sub-agent; path is a JSONL file or a "
            "directory of *.jsonl shards)"
        ),
    )
    parser.add_argument(
        "--max-think-tokens",
        type=int,
        default=128,
        help=(
            "Ceiling for generated reasoning_content (coordinator decision: "
            "128; RL-stage budgets 64/tool-turn and 128/terminal); over-limit "
            "text is truncated or the row is discarded per --think-over-limit"
        ),
    )
    parser.add_argument(
        "--think-over-limit",
        choices=("truncate", "discard"),
        default="truncate",
        help="Policy when think text exceeds --max-think-tokens",
    )
    parser.add_argument(
        "--allow-label-gaps",
        nargs="*",
        default=[],
        help=(
            "Ground-truth labels allowed to appear in val/test without any "
            "train occurrence (reviewed exceptions; recorded in the report)"
        ),
    )
    parser.add_argument(
        "--allow-any-label-gap",
        action="store_true",
        help="Waive every label gap (recorded in the report as waived)",
    )
    parser.add_argument(
        "--failed-audit",
        "--failed-audit-report",
        dest="failed_audit",
        default=None,
        help=(
            "Optional path for a failed release audit report. Existing audit "
            "paths are suffixed rather than overwritten. By default a sibling "
            "<output-dir>.failed.json path is used."
        ),
    )
    return parser.parse_args(argv)


def _finish_report(
    report: dict[str, Any],
    validation: dict[str, Any],
    staging_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Attach release validation facts and final paths before publication."""

    artifact_hashes: dict[str, str] = {}
    for split in ("train", "val", "test"):
        artifact = staging_root / f"{split}.parquet"
        if not artifact.is_file():
            raise ValueError(f"export did not produce required artifact: {artifact}")
        digest = sha256_file(artifact)
        artifact_hashes[f"{split}.parquet"] = digest
        details = report.get("splits", {}).get(split)
        if isinstance(details, dict):
            details["output_file"] = str(output_root / artifact.name)
            details["parquet_sha256"] = digest
    report["validation"] = validation
    report["release"] = {
        "status": "passed",
        "published": True,
        "output_dir": str(output_root),
        "artifacts_sha256": artifact_hashes,
    }
    (staging_root / "export_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def _think_generator(args: argparse.Namespace) -> object:
    """Resolve the configured think source to a generator instance."""

    from agent.training.sft.tool_trajectories import FileThinkGenerator, MockThinkGenerator

    if args.think_source == "mock":
        return MockThinkGenerator()
    if args.think_source.startswith("file:"):
        return FileThinkGenerator(args.think_source[len("file:") :])
    raise ValueError(
        f"unknown think source {args.think_source!r}; use 'mock' or 'file:<path>'"
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_root = None if args.output_dir is None else Path(args.output_dir)
    staging_root: Path | None = None
    phase = "prepare"
    audit_path: Path | None = None
    report: dict[str, Any] | None = None
    validation: dict[str, Any] | None = None
    try:
        if args.mode == "collect":
            if not args.collect_dir:
                raise ValueError("--mode collect requires --collect-dir")
            from agent.training.sft.tool_trajectories import collect_tool_trajectory_contexts

            task_config = _load_task_config(args.task_config, args.metadata_fields)
            corpus = _load_corpus(args.corpus)
            grading = GradingConfig.from_path(args.grading_config)
            collect_report = collect_tool_trajectory_contexts(
                args.canonical,
                args.collect_dir,
                LeafRegistry.from_path(args.registry),
                corpus=corpus,
                task_config=task_config,
                grading=grading,
                dataset=args.dataset,
                shard_size=args.shard_size,
            )
            print(json.dumps(collect_report, ensure_ascii=False, indent=2))
            return 0
        if output_root is None:
            raise ValueError("--output-dir is required in export mode")
        output_root.parent.mkdir(parents=True, exist_ok=True)
        if output_root.exists():
            if not output_root.is_dir():
                raise FileExistsError(
                    f"output target already exists and is not a directory: {output_root}"
                )
            if _is_nonempty(output_root):
                raise FileExistsError(
                    f"output directory is non-empty; refusing to overwrite existing release: {output_root}"
                )
        staging_root = Path(
            tempfile.mkdtemp(prefix=f".{output_root.name}.staging-", dir=output_root.parent)
        )
        phase = "export"
        task_config = _load_task_config(args.task_config, args.metadata_fields)
        corpus = _load_corpus(args.corpus)
        grading = GradingConfig.from_path(args.grading_config)
        grading_sha256 = None
        try:
            grading_sha256 = sha256_file(args.grading_config)
        except OSError:
            grading_sha256 = None
        report = export_tool_trajectory_dataset(
            args.canonical,
            staging_root,
            LeafRegistry.from_path(args.registry),
            corpus=corpus,
            task_config=task_config,
            grading=grading,
            dataset=args.dataset,
            think_generator=_think_generator(args),
            max_think_tokens=args.max_think_tokens,
            think_over_limit=args.think_over_limit,
            allow_label_gaps=tuple(args.allow_label_gaps),
            allow_any_label_gap=args.allow_any_label_gap,
        )
        grading_report = report.get("grading")
        if isinstance(grading_report, dict) and grading_sha256 is not None:
            grading_report["standard_sha256"] = grading_sha256
        phase = "validate"
        validation = validate_tool_trajectory_dataset(
            staging_root,
            LeafRegistry.from_path(args.registry),
            corpus=corpus,
            task_config=task_config,
            grading=grading,
            dataset=args.dataset,
            max_think_tokens=args.max_think_tokens,
        )
        if not validation.get("valid", False):
            raise ValueError("tool-trajectory export validation failed; see validation report")
        phase = "report"
        report = _finish_report(report, validation, staging_root, output_root)
        phase = "publish"
        if output_root.exists():
            if _is_nonempty(output_root):
                raise FileExistsError(
                    f"output directory became non-empty before publication: {output_root}"
                )
            output_root.rmdir()
        os.replace(staging_root, output_root)
        staging_root = None
    except Exception as exc:
        audit_path = _write_failed_audit(
            output_root,
            args.failed_audit,
            phase=phase,
            error=exc,
            staging_root=staging_root,
            export_report=report,
            validation_report=validation,
        )
        if audit_path is not None:
            print(f"failed release audit: {audit_path}", file=sys.stderr)
        print(f"export_tool_trajectory_dataset: {exc}", file=sys.stderr)
        return 2
    finally:
        if staging_root is not None and staging_root.exists():
            shutil.rmtree(staging_root, ignore_errors=True)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
