"""CLI for exporting canonical dataset records to VeRL SFT parquet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from agent.task import GradingConfig, LeafRegistry, TaskConfig
from agent.task.assets import load_corpus_categories
from agent.training.sft import export_sft_dataset


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
        help="Destination for split parquet and export_report.json",
    )
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
        grading = GradingConfig.from_path(args.grading_config) if args.grading_config else None
        report = export_sft_dataset(
            args.canonical,
            args.split_dir,
            args.output_dir,
            LeafRegistry.from_path(args.registry),
            TaskConfig.from_mapping(config_data),
            corpus=corpus,
            grading=grading,
            allow_label_gaps=tuple(args.allow_label_gaps),
            allow_any_label_gap=args.allow_any_label_gap,
        )
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"export_sft_dataset: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
