"""CLI for exporting canonical dataset records to a published VeRL SFT release.

The training export itself is deliberately kept in :mod:`agent.training.sft`.
This CLI owns the release boundary: it exports into a sibling staging
directory, validates and hashes every artifact, and only then publishes the
 directory under ``--output-dir``.  Existing non-empty release directories are
 never modified.  Failed attempts are written to a separate audit JSON file.
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
from agent.training.sft import export_sft_dataset, validate_sft_dataset


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--canonical",
        required=True,
        help="canonical <dataset>/all.json (schema v2 records with embedded splits)",
    )
    parser.add_argument(
        "--split-dir",
        default=None,
        help=(
            "Optional legacy split directory (train.json/val.json/test.json, "
            "joined by id). When omitted the embedded split fields of the "
            "schema v2 canonical records are used"
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Published destination for split parquet and export_report.json",
    )
    parser.add_argument("--registry", required=True, help="JSON leaf registry")
    parser.add_argument(
        "--corpus",
        required=True,
        help=(
            "Canonical corpus JSON (category_id/name/description/descriptions/"
            "examples); Stage 2 resolves candidates by category_id against it"
        ),
    )
    parser.add_argument(
        "--metadata-fields",
        nargs="+",
        required=True,
        help="Metadata keys visible to both stages (explicit; no implicit table_name)",
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
        default=None,
        help=(
            "Optional grading JSON (levels/descriptions/gt_field). When "
            "supplied, Stage 2 answers BOTH the classification bundle id and "
            "a sensitivity level; records lacking a level label are excluded"
        ),
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


def _is_nonempty(path: Path) -> bool:
    """Return whether ``path`` contains any directory entry."""

    try:
        next(path.iterdir())
    except StopIteration:
        return False
    return True


def _next_audit_path(output_root: Path, requested: str | None) -> Path:
    """Choose a separate audit path without destroying an earlier audit."""

    base = Path(requested) if requested else output_root.with_name(
        f"{output_root.name}.failed.json"
    )
    # Never put an audit under the release directory: that would make a
    # failed attempt look like part of an older passed release.
    try:
        base.resolve().relative_to(output_root.resolve())
    except ValueError:
        pass
    else:
        base = output_root.with_name(f"{output_root.name}.failed.json")
    base.parent.mkdir(parents=True, exist_ok=True)
    if not base.exists():
        return base
    for index in range(1, 10_000):
        candidate = base.with_name(f"{base.stem}.{index}{base.suffix}")
        if not candidate.exists():
            return candidate
    # This is practically unreachable, but avoids accidental overwrite if a
    # caller has exhausted the numbered names.
    raise RuntimeError(f"unable to choose a unique failed audit path near {base}")


def _write_failed_audit(
    output_root: Path,
    requested_audit: str | None,
    *,
    phase: str,
    error: BaseException,
    staging_root: Path | None,
    export_report: dict[str, Any] | None = None,
    validation_report: dict[str, Any] | None = None,
) -> Path | None:
    """Persist a failed attempt without touching the requested release."""

    try:
        path = _next_audit_path(output_root, requested_audit)
        payload = {
            "status": "failed",
            "phase": phase,
            "error": str(error),
            "error_type": type(error).__name__,
            "output_dir": str(output_root),
            "staging_dir": str(staging_root) if staging_root is not None else None,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "export_report": export_report,
            "validation": validation_report,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path
    except (OSError, RuntimeError) as audit_error:
        print(f"could not persist failed audit report: {audit_error}", file=sys.stderr)
        return None


def _load_task_config(path: str | None, metadata_fields: list[str]) -> TaskConfig:
    config_data: dict[str, Any] = {}
    if path:
        with Path(path).open(encoding="utf-8") as handle:
            config_data = json.load(handle)
        if not isinstance(config_data, dict):
            raise ValueError("task config must be a JSON object")
    config_data["metadata_fields"] = metadata_fields
    return TaskConfig.from_mapping(config_data)


def _load_corpus(path: str) -> dict[str, Any]:
    return {
        category.category_id: category
        for category in load_corpus_categories(path)
    }


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


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_root = Path(args.output_dir)
    staging_root: Path | None = None
    phase = "prepare"
    audit_path: Path | None = None
    report: dict[str, Any] | None = None
    validation: dict[str, Any] | None = None
    try:
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
        # The staging directory is a sibling so os.replace is a same-filesystem
        # publication. It is hidden to make accidental consumption impossible.
        staging_root = Path(
            tempfile.mkdtemp(prefix=f".{output_root.name}.staging-", dir=output_root.parent)
        )
        phase = "export"
        task_config = _load_task_config(args.task_config, args.metadata_fields)
        corpus = _load_corpus(args.corpus)
        grading = GradingConfig.from_path(args.grading_config) if args.grading_config else None
        grading_sha256 = sha256_file(args.grading_config) if args.grading_config else None
        report = export_sft_dataset(
            args.canonical,
            args.split_dir,
            staging_root,
            LeafRegistry.from_path(args.registry),
            task_config,
            corpus=corpus,
            grading=grading,
            allow_label_gaps=tuple(args.allow_label_gaps),
            allow_any_label_gap=args.allow_any_label_gap,
        )
        if grading is not None:
            grading_report = report.get("grading")
            if isinstance(grading_report, dict):
                grading_report["standard_sha256"] = grading_sha256
        phase = "validate"
        validation = validate_sft_dataset(
            staging_root,
            LeafRegistry.from_path(args.registry),
            task_config,
            corpus=corpus,
            grading=grading,
        )
        if not validation.get("valid", False):
            raise ValueError("SFT export validation failed; see validation report")
        phase = "report"
        report = _finish_report(report, validation, staging_root, output_root)
        phase = "publish"
        # An empty pre-existing directory is not a release. Remove it only
        # after all checks are complete; a non-empty old release was rejected
        # before staging and remains byte-for-byte untouched.
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
        print(f"export_sft_dataset: {exc}", file=sys.stderr)
        return 2
    finally:
        if staging_root is not None and staging_root.exists():
            shutil.rmtree(staging_root, ignore_errors=True)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
