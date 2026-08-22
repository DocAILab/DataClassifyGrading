"""CLI for exporting canonical dataset records to the VeRL RL parquet.

Output schema (verl v0.8.0 RL five-field contract):
    data_source / prompt / ability / reward_model / extra_info
Prompt carries system+user only; the assistant gold response is NOT stored.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from agent.task import LeafRegistry, TaskConfig
from agent.task.assets import load_corpus_categories
from agent.training.rl import export_rl_dataset


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--canonical",
        required=True,
        help="data/canonical/<dataset>/all.json (canonical dataset records)",
    )
    parser.add_argument(
        "--split-dir",
        required=True,
        help="Directory containing train.json, val.json, test.json (split boundaries by id)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Destination for split parquet and export_report.json",
    )
    parser.add_argument("--dataset", required=True, help="Dataset name (goes into data_source)")
    parser.add_argument("--registry", required=True, help="JSON leaf registry")
    parser.add_argument(
        "--corpus",
        required=True,
        help="Canonical corpus JSON (category_id/name/description/descriptions/"
        "examples); Stage 2 resolves candidates by category_id against it",
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
        corpus = {
            category.category_id: category
            for category in load_corpus_categories(args.corpus)
        }
        report = export_rl_dataset(
            args.canonical,
            args.split_dir,
            args.output_dir,
            args.dataset,
            LeafRegistry.from_path(args.registry),
            TaskConfig.from_mapping(config_data),
            corpus=corpus,
        )
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"export_rl_dataset: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
