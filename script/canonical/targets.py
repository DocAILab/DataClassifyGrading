"""Resolve every dataset record to a canonical target (stage 3B).

Usage:
    python -m script.canonical.targets [--dataset finance] [--datasets ...] [--overwrite]

Input:   data/processed/<dataset>/all.json (normalized records)
Writes into data/canonical/<dataset>/:
    all.json
        every input record unchanged (classification untouched) plus
        "resolution_status" and, for resolved records, "target".
    resolution_report.json
        status counts, unresolved details, registry facts, input sha256.

The LeafRegistry is the final constraint: a resolver target only counts as
resolved when its category_id belongs to the registry. All datasets are built
before anything is written (fail-fast), every output is written exactly once,
and the CLI refuses to overwrite without --overwrite.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agent.task.canonical_dataset import (  # noqa: E402
    CanonicalDatasetResult,
    prepare_canonical_dataset,
    write_canonical_dataset,
)

DEFAULT_DATASETS = ("shougang", "infra", "finance", "pers_info")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed",
        help="Where processed records live: processed-dir/<dataset>/all.json",
    )
    parser.add_argument(
        "--canonical-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "canonical",
        help="Where the canonical contract is written: canonical-dir/<dataset>/",
    )

    parser.add_argument("--registry-dir", type=Path, default=PROJECT_ROOT / "cfg" / "task" / "registry")
    parser.add_argument("--corpus-dir", type=Path, default=PROJECT_ROOT / "cfg" / "task" / "corpus")
    parser.add_argument("--dataset", type=str, choices=list(DEFAULT_DATASETS))
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DEFAULT_DATASETS),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    if args.dataset and args.datasets:
        parser.error("use either --dataset or --datasets, not both")
    datasets = (
        [args.dataset]
        if args.dataset
        else (args.datasets or list(DEFAULT_DATASETS))
    )

    # 1. fail fast before building/writing anything
    if not args.overwrite:
        existing = [
            Path(args.canonical_dir) / dataset / name
            for dataset in datasets
            for name in ("all.json", "resolution_report.json")
            if (Path(args.canonical_dir) / dataset / name).exists()
        ]
        if existing:
            raise FileExistsError(
                "refusing to overwrite existing canonical output(s): "
                + ", ".join(str(path) for path in existing)
                + " (pass --overwrite to replace them)"
            )

    # 2. prepare EVERY selected dataset (pure computation, no writes): if any
    #    dataset fails to load/validate/resolve, no output is produced at all
    prepared: list[tuple[CanonicalDatasetResult, list[dict]]] = []
    for dataset in datasets:
        prepared.append(
            prepare_canonical_dataset(
                dataset,
                processed_dir=args.processed_dir,
                canonical_dir=args.canonical_dir,
                registry_dir=args.registry_dir,
                corpus_dir=args.corpus_dir,
            )
        )

    # 3. all prepared successfully -> write every output exactly once
    for result, output_records in prepared:
        write_canonical_dataset(result, output_records)
        print(
            f"{result.dataset}: input={result.input_records} "
            f"status={dict(result.status_counts)} "
            f"registry={result.registry_size}"
        )
        print(f"  wrote: {result.output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
