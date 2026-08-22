"""CLI for validating VeRL SFT parquet and explicit contracts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from agent.task import LeafRegistry, TaskConfig
from agent.task.assets import load_corpus_categories
from agent.training.sft import validate_sft_dataset


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument(
        "--corpus",
        required=True,
        help="Canonical corpus JSON for Stage 2 prompt checks (required; no fallback)",
    )
    parser.add_argument("--metadata-fields", nargs="+", required=True)
    parser.add_argument("--task-config")
    parser.add_argument("--report", help="Optional path for the structured validation report")
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
        corpus = (
            {
                category.category_id: category
                for category in load_corpus_categories(args.corpus)
            }
            if args.corpus
            else None
        )
        report = validate_sft_dataset(
            args.dataset_dir,
            LeafRegistry.from_path(args.registry),
            TaskConfig.from_mapping(config_data),
            corpus=corpus,
        )
        if args.report:
            Path(args.report).parent.mkdir(parents=True, exist_ok=True)
            Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["valid"] else 1
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"validate_sft_dataset: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
