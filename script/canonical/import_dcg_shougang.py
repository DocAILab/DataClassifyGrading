"""Prepare the official DCG shougang-5415 split for baseline evaluation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile
import uuid
from typing import Any

from agent.task.dcg_shougang import PATH_FIELDS, prepare_shougang_5415


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--no-strict-counts",
        action="store_true",
        help="Disable the real-data split and category-count assertions",
    )
    parser.add_argument(
        "--identity-mode",
        choices=("level4", "full-path"),
        default="level4",
        help="Category identity: published level4 (193 classes) or full path (203 classes)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Atomically replace an existing non-empty runtime release",
    )
    return parser.parse_args(argv)


def _nonempty(path: Path) -> bool:
    return any(path.iterdir())


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _write_staging(staging: Path, prepared) -> None:
    canonical_root = staging / "canonical" / "shougang"
    _write_json(canonical_root / "all.json", list(prepared.canonical_records))
    _write_json(canonical_root / "import_report.json", prepared.report)
    _write_json(
        staging / "registries" / "shougang.registry.json", prepared.registry
    )
    _write_json(staging / "corpus" / "shougang.corpus.json", prepared.corpus)
    _write_json(staging / "cfg" / "datasets.json", prepared.dataset_config)
    _write_json(staging / "cfg" / "task.json", prepared.task_config)
    _write_json(staging / "cfg" / "grading.json", prepared.grading_config)


def _publish(staging: Path, output: Path, *, overwrite: bool) -> None:
    backup: Path | None = None
    if output.exists():
        if not output.is_dir():
            raise FileExistsError(f"output path is not a directory: {output}")
        if _nonempty(output):
            if not overwrite:
                raise FileExistsError(
                    f"output directory is non-empty; refusing to replace it: {output}"
                )
            backup = output.with_name(f".{output.name}.backup-{uuid.uuid4().hex}")
            os.replace(output, backup)
        else:
            output.rmdir()
    try:
        os.replace(staging, output)
    except BaseException:
        if backup is not None and backup.exists() and not output.exists():
            os.replace(backup, output)
        raise
    if backup is not None:
        shutil.rmtree(backup)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output = Path(args.output_dir)
    if output.exists() and output.is_dir() and _nonempty(output) and not args.overwrite:
        raise FileExistsError(
            f"output directory is non-empty; refusing to replace it: {output}"
        )
    if output.exists() and not output.is_dir():
        raise FileExistsError(f"output path is not a directory: {output}")

    prepared = prepare_shougang_5415(
        args.source_dir,
        strict_counts=not args.no_strict_counts,
        identity_fields=(PATH_FIELDS if args.identity_mode == "full-path" else ("level_4",)),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent)
    )
    try:
        _write_staging(staging, prepared)
        _publish(staging, output, overwrite=args.overwrite)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    print(json.dumps(prepared.report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
