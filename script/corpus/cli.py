"""Command-line interface for building retrieval corpora."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from .builder import (
    DEFAULT_CONTENT_COLUMN,
    DEFAULT_LABEL_COLUMN,
    build_corpus,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def _dataset_name(value: str) -> str:
    dataset = value.strip().lower()
    if not DATASET_PATTERN.fullmatch(dataset):
        raise argparse.ArgumentTypeError(
            "dataset must contain only letters, numbers, underscores, and hyphens"
        )
    return dataset


def _default_input(dataset: str) -> Path:
    raw_dir = PROJECT_ROOT / "data" / "raw"
    candidates = [
        raw_dir / f"{dataset}-content.xlsx",
        raw_dir / f"{dataset}-content.csv",
    ]
    existing = [path for path in candidates if path.is_file()]
    if len(existing) == 1:
        return existing[0]
    if len(existing) > 1:
        raise ValueError(
            "Multiple corpus sources found: "
            + ", ".join(str(path) for path in existing)
            + ". Select one with --input."
        )
    raise FileNotFoundError(
        f"No corpus source found for '{dataset}'. Expected "
        + " or ".join(str(path) for path in candidates)
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build retrieval corpus JSON from classification guides."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="Generate corpus and coverage report.")
    build.add_argument("--dataset", required=True, type=_dataset_name)
    build.add_argument("--input", type=Path)
    build.add_argument("--processed", type=Path)
    build.add_argument("--output-dir", type=Path)
    build.add_argument("--label-column", default=DEFAULT_LABEL_COLUMN)
    build.add_argument("--content-column", default=DEFAULT_CONTENT_COLUMN)
    build.add_argument(
        "--missing-policy",
        choices=("error", "skip"),
        default="skip",
        help="How to handle rows missing a label or content (default: skip).",
    )
    build.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        source = args.input or _default_input(args.dataset)
        processed = args.processed
        if processed is None:
            candidate = PROJECT_ROOT / "data" / "processed" / args.dataset / "all.json"
            processed = candidate if candidate.is_file() else None
        output_dir = (
            args.output_dir
            or PROJECT_ROOT / "data" / "corpus" / args.dataset
        )
        report = build_corpus(
            args.dataset,
            source,
            output_dir,
            processed_file=processed,
            label_column=args.label_column,
            content_column=args.content_column,
            missing_policy=args.missing_policy,
            overwrite=args.overwrite,
            project_root=PROJECT_ROOT,
        )
        summary = {
            "input_rows": report["input_rows"],
            "exported_documents": report["exported_documents"],
            "unique_level_4": report["corpus_statistics"]["unique_level_4"],
            "processed_label_overlap": report["processed_label_coverage"]["overlap"],
        }
        print(json.dumps(summary, ensure_ascii=False))
        print(f"Corpus saved to: {Path(output_dir).resolve()}")
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
