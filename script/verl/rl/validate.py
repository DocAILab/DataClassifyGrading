"""CLI for validating exported VeRL RL parquet files.

Rebuilds both stage prompts from the canonical registry + corpus and
compares them against the stored rows, so a drifted prompt contract is a
validation failure, not a silent training surprise.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from agent.task import LeafRegistry, TaskConfig
from agent.task.canonical_dataset import load_corpus_categories
from agent.training.rl import validate_rl_dataset


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, help="Directory with train/val/test.parquet")
    parser.add_argument("--dataset", required=True, help="Dataset name (must match exported data_source)")
    parser.add_argument("--registry", required=True, help="JSON leaf registry")
    parser.add_argument(
        "--corpus",
        required=True,
        help="Canonical corpus JSON; Stage 2 prompts are rebuilt from it",
    )
    parser.add_argument(
        "--metadata-fields",
        nargs="+",
        required=True,
        help="Metadata keys visible to both stages (must match export)",
    )
    parser.add_argument(
        "--task-config",
        help="Optional JSON task config (task_name); CLI metadata_fields take precedence",
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
        report = validate_rl_dataset(
            args.dataset_dir,
            args.dataset,
            LeafRegistry.from_path(args.registry),
            TaskConfig.from_mapping(config_data),
            corpus=corpus,
        )
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"validate_rl_dataset: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
