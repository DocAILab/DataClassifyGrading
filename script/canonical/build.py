"""Build the canonical contract layer from processed records.

Reads ``<processed-dir>/<dataset>/all.json`` for each requested dataset,
resolves every record against the runtime-supplied LeafRegistry according to
the runtime-loaded DatasetConfig, and writes atomically:

    <canonical-dir>/<dataset>/all.json                (schema v2 records)
    <canonical-dir>/<dataset>/resolution_report.json  (deterministic report)

Cross-dataset fail-fast: every selected dataset is prepared before anything
is written, so a failure in one dataset leaves no partial outputs.

All inputs are explicit local paths; production registries/corpora/configs
are runtime-local assets and never searched implicitly.

Example:
    python -m script.canonical.build \\
        --processed-dir <local>/processed \\
        --canonical-dir <local>/canonical \\
        --config-file <local>/datasets.config.json \\
        --registry-dir <local>/registries \\
        --corpus-dir <local>/corpora \\
        --dataset finance --dataset pers_info --overwrite
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from agent.task import (
    LeafRegistry,
    load_dataset_configs,
)
from agent.task.canonical_builder import (
    load_corpus_categories_file,
    prepare_canonical_dataset,
    write_canonical_dataset,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--processed-dir", required=True,
                        help="Directory containing <dataset>/all.json inputs")
    parser.add_argument("--canonical-dir", required=True,
                        help="Destination root; <dataset>/all.json is written inside")
    parser.add_argument("--config-file", required=True,
                        help="Dataset config JSON (object of datasets or array)")
    parser.add_argument("--registry-dir", required=True,
                        help="Directory containing <registry-name>.registry.json")
    parser.add_argument("--corpus-dir",
                        help="Directory containing <dataset>.corpus.json "
                             "(required by datasets with id_strategy=code)")
    parser.add_argument("--dataset", action="append", dest="datasets",
                        required=True,
                        help="Dataset name to build; repeatable")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace existing canonical outputs")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configs = load_dataset_configs(args.config_file)
    unknown = sorted(set(args.datasets) - set(configs))
    if unknown:
        raise SystemExit(
            f"datasets missing from config file {args.config_file}: {unknown}"
        )
    if not args.corpus_dir and any(
        configs[name].id_strategy == "code" for name in args.datasets
    ):
        raise SystemExit("--corpus-dir is required for id_strategy=code datasets")

    prepared_batches: list[tuple[str, object, list[dict]]] = []
    for dataset in args.datasets:
        config = configs[dataset]
        effective_registry = config.registry_source or dataset
        registry_file = Path(args.registry_dir) / f"{effective_registry}.registry.json"
        if not registry_file.is_file():
            raise SystemExit(f"registry not found: {registry_file}")
        registry = LeafRegistry.from_path(registry_file)

        corpus_categories = None
        if config.id_strategy == "code":
            corpus_file = Path(args.corpus_dir) / f"{dataset}.corpus.json"
            if not corpus_file.is_file():
                raise SystemExit(f"corpus not found: {corpus_file}")
            corpus_categories = load_corpus_categories_file(corpus_file)

        processed_file = Path(args.processed_dir) / dataset / "all.json"
        output_root = Path(args.canonical_dir)
        result, records = prepare_canonical_dataset(
            dataset,
            processed_file=processed_file,
            output_file=output_root / dataset / "all.json",
            config=config,
            registry=registry,
            registry_file=registry_file,
            corpus_categories=corpus_categories,
        )
        prepared_batches.append((dataset, result, records))

    # everything prepared: now write (cross-dataset fail-fast)
    summary = {}
    for dataset, result, records in prepared_batches:
        write_canonical_dataset(result, records, overwrite=args.overwrite)
        summary[dataset] = {
            "input_records": result.input_records,
            "status_counts": dict(result.status_counts),
        }
    print(json.dumps({"status": "ok", "datasets": summary}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
