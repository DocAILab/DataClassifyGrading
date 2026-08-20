"""CLI for exporting train/val/test JSON to VeRL SFT parquet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from agent.task import LeafRegistry, TaskConfig
from method.sft import export_sft_dataset


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing train.json, val.json, test.json",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Destination for split parquet and export_report.json",
    )
    parser.add_argument("--registry", required=True, help="JSON leaf registry")
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
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config_data = {}
        if args.task_config:
            with Path(args.task_config).open(encoding="utf-8") as handle:
                config_data = json.load(handle)
            if not isinstance(config_data, dict):
                raise ValueError("task config must be a JSON object")
        config_data["metadata_fields"] = args.metadata_fields
        report = export_sft_dataset(
            args.input_dir,
            args.output_dir,
            LeafRegistry.from_path(args.registry),
            TaskConfig.from_mapping(config_data),
            splits=args.splits,
        )
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"export_sft_dataset: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
