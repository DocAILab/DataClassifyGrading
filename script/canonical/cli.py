"""Build canonical corpus JSON + registry JSON for the four datasets.

Usage:
    python -m script.canonical.cli [--data-dir data] [--out-dir cfg/task]
                                   [--datasets finance infra pers_info shougang]
                                   [--overwrite]

Writes per dataset:
    <out-dir>/corpus/<dataset>.corpus.json     canonical corpus (CorpusCategory)
    <out-dir>/registry/<dataset>.registry.json LeafRegistry.from_path-compatible

Sources:
- shougang / infra: data/knowledge/standards_map/guanji_dict.json
- finance:          data/knowledge/standards_map/financial_standards_dict.json
- pers_info:        data/<dataset>/all.json (dataset universe, no standard exists)

All builds and coverage diagnostics are computed before anything is written;
every output file is written exactly once. Artifact sources are
repo-relative logical paths. The script never modifies data/ or src/ and
never repairs labels.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agent.task import BUILTIN_DATASET_CONFIGS  # noqa: E402
from agent.task.canonical_corpus import (  # noqa: E402
    BuildReport,
    build_finance_corpus,
    build_infra_corpus,
    build_pers_info_corpus,
    build_shougang_corpus,
    compute_dataset_id_coverage,
    corpus_to_mapping,
    registry_to_mapping,
)
from agent.task.identity import code_leaf_map  # noqa: E402

DEFAULT_DATASETS = ("shougang", "infra", "finance", "pers_info")


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _load_records(data_dir: Path, dataset: str) -> list[dict[str, Any]]:
    records = _load_json(data_dir / dataset / "all.json")
    if not isinstance(records, list):
        raise ValueError(f"{data_dir / dataset / 'all.json'} must be a JSON list")
    return records


def _builders() -> dict[str, Callable[[Path, Path, str], tuple[list, BuildReport]]]:
    def pers(data_dir: Path, _standards: Path, dataset: str):
        return build_pers_info_corpus(_load_records(data_dir, dataset), dataset)

    return {
        "shougang": lambda data_dir, standards, dataset: build_shougang_corpus(
            standards / "guanji_dict.json", dataset
        ),
        "infra": lambda data_dir, standards, dataset: build_infra_corpus(
            standards / "guanji_dict.json", dataset
        ),
        "finance": lambda data_dir, standards, dataset: build_finance_corpus(
            standards / "financial_standards_dict.json", dataset
        ),
        "pers_info": pers,
    }


def _annotate_coverage(
    dataset: str,
    categories: list,
    report: BuildReport,
    data_dir: Path,
) -> None:
    """Read-only registry vs resolver-ID coverage; mutates the report only."""
    records_path = data_dir / dataset / "all.json"
    if not records_path.is_file():
        report.dataset_id_coverage = {"available": False}
        return
    records = _load_records(data_dir, dataset)
    config = BUILTIN_DATASET_CONFIGS[dataset]
    code_map = (
        code_leaf_map(categories)
        if config.id_strategy == "code"
        else {}
    )
    registry_ids = {category.category_id for category in categories}
    report.dataset_id_coverage = {
        "available": True,
        **compute_dataset_id_coverage(dataset, registry_ids, records, config, code_map),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "cfg" / "task")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        choices=list(DEFAULT_DATASETS),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    data_dir: Path = args.data_dir
    out_dir: Path = args.out_dir
    standards = data_dir / "knowledge" / "standards_map"
    builders = _builders()

    # 1. build everything and compute coverage (no writes yet)
    pending: list[tuple[Path, Any]] = []
    for dataset in args.datasets:
        categories, report = builders[dataset](data_dir, standards, dataset)
        _annotate_coverage(dataset, categories, report, data_dir)
        pending.append(
            (
                out_dir / "corpus" / f"{dataset}.corpus.json",
                corpus_to_mapping(dataset, report.source, report.id_strategy,
                                  categories, report),
            )
        )
        pending.append(
            (
                out_dir / "registry" / f"{dataset}.registry.json",
                registry_to_mapping(categories),
            )
        )

    # 2. fail fast if anything exists before writing anything
    if not args.overwrite:
        existing = [path for path, _ in pending if path.exists()]
        if existing:
            raise FileExistsError(
                "refusing to overwrite existing output(s): "
                + ", ".join(str(path) for path in existing)
                + " (pass --overwrite to replace them)"
            )

    # 3. write each output exactly once
    for path, data in pending:
        _write_json(path, data)

    print(
        "wrote:",
        ", ".join(
            _display_path(path) for path, _ in pending
        ),
    )
    return 0


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
