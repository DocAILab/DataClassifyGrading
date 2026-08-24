"""Generic CLI for tabular preprocessing and processed-layer splitting.

Rebuilt without any dataset-specific defaults: every input, mapping, and
output path is explicit. Production data stays runtime-local.

Examples:
    python -m script.preprocessing.cli preprocess \\
        --input <local>.xlsx --mapping mappings/finance.json \\
        --output <local>/processed/<ds>/all.json --dataset <ds>

    python -m script.preprocessing.cli split \\
        --input <local>/processed/<ds>/all.json \\
        --output-dir <local>/processed/<ds> \\
        --split-type group --group-key metadata.table_name
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from script.preprocessing.processor import preprocess
from script.preprocessing.split import split_dataset


def _load_rewrite_rules(path: str | None) -> list[dict] | None:
    if path is None:
        return None
    source = Path(path)
    with source.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(
            f"rewrite rules must be a JSON array of objects: {source}"
        )
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    pre = subparsers.add_parser("preprocess", help="CSV/XLSX -> normalized JSON")
    pre.add_argument("--input", required=True)
    pre.add_argument("--mapping", required=True)
    pre.add_argument("--output", required=True)
    pre.add_argument("--dataset", required=True,
                     help="Dataset name; part of the stable sample id")
    pre.add_argument("--overwrite", action="store_true")
    pre.add_argument("--missing-field-policy", choices=("error", "skip"), default="error")
    pre.add_argument("--strip-trailing-codes", action="store_true",
                     help="Legacy behavior: remove bracket-style label suffixes "
                          "(default: keep them and record them in label_notes)")
    pre.add_argument("--rewrite-rules",
                     help="JSON array of {match, replace, field} label rules")

    spl = subparsers.add_parser("split", help="Normalized JSON -> train/val/test")
    spl.add_argument("--input", required=True)
    spl.add_argument("--output-dir", required=True)
    spl.add_argument("--split-type", choices=("random", "group"), default="random")
    spl.add_argument("--group-key", help="e.g. metadata.table_name")
    spl.add_argument("--train-ratio", type=float, default=0.8)
    spl.add_argument("--val-ratio", type=float, default=0.1)
    spl.add_argument("--test-ratio", type=float, default=0.1)
    spl.add_argument("--seed", type=int, default=42)
    spl.add_argument("--missing-group-policy", choices=("error", "skip"), default="error")
    spl.add_argument("--overwrite", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "preprocess":
        records = preprocess(
            args.input,
            args.mapping,
            args.output,
            dataset=args.dataset,
            overwrite=args.overwrite,
            missing_field_policy=args.missing_field_policy,
            strip_trailing_codes=args.strip_trailing_codes,
            rewrite_rules=_load_rewrite_rules(args.rewrite_rules),
        )
        print(f"records: {len(records)} output: {args.output}")
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
        print(json.dumps({"sizes": report["sizes"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
