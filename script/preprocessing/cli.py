"""Command-line entry point for TransClass preprocessing."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from .processor import preprocess
from .split import split_dataset


MAPPINGS_DIR = Path(__file__).resolve().parent / "mappings"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def _normalize_dataset_name(value: str) -> str:
    normalized = value.strip().lower()
    if not DATASET_NAME_PATTERN.fullmatch(normalized):
        raise ValueError(
            "dataset must contain only letters, numbers, underscores, and hyphens"
        )
    return normalized


def dataset_name(value: str) -> str:
    """Argparse adapter for safe dataset names."""
    try:
        return _normalize_dataset_name(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def discover_datasets() -> tuple[str, ...]:
    """Discover datasets from JSON files in the mappings directory."""
    return tuple(sorted(
        path.stem
        for path in MAPPINGS_DIR.glob("*.json")
        if DATASET_NAME_PATTERN.fullmatch(path.stem)
    ))


def resolve_mapping(dataset: str | None, mapping: Path | None) -> Path:
    """Resolve either a built-in dataset mapping or a custom mapping path."""
    if mapping is not None:
        return mapping
    if dataset is None:
        raise ValueError("either --dataset or --mapping is required")
    dataset = _normalize_dataset_name(dataset)
    mapping_path = MAPPINGS_DIR / f"{dataset}.json"
    if not mapping_path.is_file():
        raise FileNotFoundError(
            f"Mapping for dataset '{dataset}' not found: {mapping_path}. "
            "Add that file or pass --mapping."
        )
    return mapping_path


def resolve_default_input(dataset: str) -> Path:
    """Find a conventional CSV/XLSX raw input for a dataset."""
    dataset = _normalize_dataset_name(dataset)
    raw_dir = PROJECT_ROOT / "data" / "raw"
    # The plural form keeps compatibility with files such as <dataset>s.xlsx.
    stems = (dataset, f"{dataset}s")
    candidates = [
        raw_dir / f"{stem}{suffix}"
        for stem in stems
        for suffix in (".xlsx", ".csv")
    ]
    existing = [path for path in candidates if path.is_file()]
    if len(existing) == 1:
        return existing[0]
    if len(existing) > 1:
        raise ValueError(
            f"Multiple raw files found for '{dataset}': "
            + ", ".join(str(path) for path in existing)
            + ". Select one with --input."
        )
    raise FileNotFoundError(
        f"No raw file found for '{dataset}' in {raw_dir}. Expected one of: "
        + ", ".join(path.name for path in candidates)
        + "."
    )


def _ratio(value: str) -> float:
    try:
        ratio = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ratio must be a number") from exc
    if not 0 <= ratio <= 1:
        raise argparse.ArgumentTypeError("ratio must be between 0 and 1")
    return ratio


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preprocess and split TransClass datasets without config files."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser(
        "list-datasets",
        help="List datasets discovered from preprocessing/mappings/*.json.",
    )

    preprocess_parser = commands.add_parser(
        "preprocess", help="Convert CSV/XLSX metadata to normalized JSON."
    )
    preprocess_parser.add_argument("--input", required=True, type=Path)
    mapping_source = preprocess_parser.add_mutually_exclusive_group(required=True)
    mapping_source.add_argument(
        "--dataset",
        type=dataset_name,
        help="Use mappings/<dataset>.json.",
    )
    mapping_source.add_argument(
        "--mapping",
        type=Path,
        help="Use a custom mapping JSON file.",
    )
    preprocess_parser.add_argument("--output", required=True, type=Path)
    preprocess_parser.add_argument(
        "--missing-field-policy",
        choices=("error", "skip"),
        default="error",
        help="How to handle rows without field_name (default: error).",
    )
    preprocess_parser.add_argument(
        "--overwrite", action="store_true", help="Replace an existing output file."
    )

    prepare_parser = commands.add_parser(
        "prepare",
        help="Discover, preprocess, and split a dataset in one command.",
    )
    prepare_parser.add_argument(
        "--dataset", required=True, type=dataset_name
    )
    prepare_parser.add_argument(
        "--input",
        type=Path,
        help="Raw CSV/XLSX path. Defaults to data/raw/<dataset>.",
    )
    prepare_parser.add_argument(
        "--mapping",
        type=Path,
        help="Override the mapping bundled for the selected dataset.",
    )
    prepare_parser.add_argument(
        "--output-dir",
        type=Path,
        help="Defaults to data/processed/<dataset>.",
    )
    prepare_parser.add_argument(
        "--split-type",
        choices=("random", "group"),
        default="group",
        help="Defaults to group, which keeps related records together.",
    )
    prepare_parser.add_argument(
        "--group-key",
        default="metadata.table_name",
        help=(
            "Nested grouping field used for group splitting "
            "(default: metadata.table_name)."
        ),
    )
    prepare_parser.add_argument(
        "--missing-group-policy",
        choices=("error", "skip"),
        default="skip",
        help="How to handle empty group keys (default: skip).",
    )
    prepare_parser.add_argument("--train-ratio", type=_ratio, default=0.8)
    prepare_parser.add_argument("--val-ratio", type=_ratio, default=0.1)
    prepare_parser.add_argument("--test-ratio", type=_ratio, default=0.1)
    prepare_parser.add_argument("--seed", type=int, default=42)
    prepare_parser.add_argument(
        "--missing-field-policy",
        choices=("error", "skip"),
        default="skip",
        help="How to handle table-level rows without field_name (default: skip).",
    )
    prepare_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace all.json and existing split outputs.",
    )

    split_parser = commands.add_parser(
        "split", help="Create train.json, val.json, test.json, and a report."
    )
    split_parser.add_argument("--input", required=True, type=Path)
    split_parser.add_argument("--output-dir", required=True, type=Path)
    split_parser.add_argument(
        "--type", choices=("random", "group"), default="random", dest="split_type"
    )
    split_parser.add_argument("--train-ratio", type=_ratio, default=0.8)
    split_parser.add_argument("--val-ratio", type=_ratio, default=0.1)
    split_parser.add_argument("--test-ratio", type=_ratio, default=0.1)
    split_parser.add_argument("--seed", type=int, default=42)
    split_parser.add_argument(
        "--group-key", help="Nested key for group splitting, e.g. metadata.table_name."
    )
    split_parser.add_argument(
        "--missing-group-policy",
        choices=("error", "skip"),
        default="error",
        help="How to handle empty group keys (default: error).",
    )
    split_parser.add_argument(
        "--overwrite", action="store_true", help="Replace existing split outputs."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "list-datasets":
            datasets = discover_datasets()
            if datasets:
                print("\n".join(datasets))
            else:
                print(f"No mappings found in {MAPPINGS_DIR}")
        elif args.command == "preprocess":
            mapping = resolve_mapping(args.dataset, args.mapping)
            result = preprocess(
                args.input,
                mapping,
                args.output,
                overwrite=args.overwrite,
                missing_field_policy=args.missing_field_policy,
            )
            print(f"Preprocessed {len(result)} records -> {args.output}")
        elif args.command == "prepare":
            input_file = args.input or resolve_default_input(args.dataset)
            mapping = resolve_mapping(args.dataset, args.mapping)
            output_dir = (
                args.output_dir
                or PROJECT_ROOT / "data" / "processed" / args.dataset
            )
            all_file = output_dir / "all.json"
            result = preprocess(
                input_file,
                mapping,
                all_file,
                overwrite=args.overwrite,
                missing_field_policy=args.missing_field_policy,
            )
            print(f"Preprocessed {len(result)} records -> {all_file}")
            report = split_dataset(
                all_file,
                output_dir,
                split_type=args.split_type,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                random_seed=args.seed,
                group_key=(
                    args.group_key if args.split_type == "group" else None
                ),
                missing_group_policy=args.missing_group_policy,
                overwrite=args.overwrite,
            )
            print(json.dumps(report["sizes"], ensure_ascii=False))
            print(
                f"Prepared dataset in {output_dir}; "
                f"warnings: {len(report['warnings'])}"
            )
        else:
            report = split_dataset(
                args.input,
                args.output_dir,
                split_type=args.split_type,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                random_seed=args.seed,
                group_key=args.group_key,
                missing_group_policy=args.missing_group_policy,
                overwrite=args.overwrite,
            )
            print(json.dumps(report["sizes"], ensure_ascii=False))
            print(
                f"Saved split files to {args.output_dir}; "
                f"warnings: {len(report['warnings'])}"
            )
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
