"""Build canonical corpus JSON + registry JSON for the four datasets.

Usage:
    python -m script.canonical.cli [--data-dir data] [--out-dir cfg/task]
                                   [--standard-dir data/standards]
                                   [--datasets finance infra pers_info shougang]
                                   [--overwrite]

Writes per dataset:
    <out-dir>/corpus/<dataset>.corpus.json     canonical corpus (CorpusCategory)
    <out-dir>/registry/<dataset>.registry.json LeafRegistry.from_path-compatible

Sources (Phase-2 chain):  raw standard
                          → CanonicalStandard (Phase 1, data/standards)
                          → LeafRegistry + Corpus
- finance:  data/standards/finance.standard.json  (237 entries → 233 categories)
- shougang: data/standards/shougang.standard.json  (234 entries; B3-6 reported,
            not silently added/dropped)
- infra:    shared shougang standard (registry_source=shougang)
- pers_info: no standard → dataset-derived fallback (data/processed/all.json)

The legacy standards_map digests (financial_standards_dict.json /
guanji_dict.json) are NOT read by this formal build (audit-only).

To keep Stage-1/Stage-2 prompt behavior byte-identical, registry/corpus order
follows the FIRST SOURCE ROW of each category and registry path keeps the
historical Stage-1 values; universe membership comes from the EXPLICIT
projection policy (DatasetConfig.projection_excluded_category_ids — shougang
excludes B3-6; infra inherits), and excluded categories are reported. The
legacy registry/standards_map are never read by this build (audit/parity only).

All builds and coverage diagnostics are computed before anything is written;
every output file is written exactly once. Artifact sources are repo-relative
logical paths. The script never modifies data/ or src/ and never repairs
labels, and never cascades standard_data_level/annotations into prompts.
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
    build_from_standard,
    build_pers_info_corpus,
    compute_dataset_id_coverage,
    corpus_to_mapping,
    registry_to_mapping,
)
from agent.task.identity import code_leaf_map  # noqa: E402

DEFAULT_DATASETS = ("shougang", "infra", "finance", "pers_info")

# dataset -> canonical standard key under --standard-dir (None = no standard)
STANDARD_DATASETS = {
    "finance": "finance",
    "shougang": "shougang",
    "infra": "shougang",
    "pers_info": None,
}


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _load_records(data_dir: Path, dataset: str) -> list[dict[str, Any]]:
    records = _load_json(data_dir / "processed" / dataset / "all.json")
    if not isinstance(records, list):
        raise ValueError(
            f"{data_dir / 'processed' / dataset / 'all.json'} must be a JSON list"
        )
    return records


def _load_standard(path: Path):
    from agent.standards.contracts import CanonicalStandard

    if not path.is_file():
        raise FileNotFoundError(
            f"canonical standard not found: {path} — build it first via "
            "`python -m script.standard.cli` (restore/verify data/raw per "
            "docs/design/phase1_canonical_standard.md, then regenerate)"
        )
    return CanonicalStandard.from_mapping(_load_json(path))


def _builders(
    standard_dir: Path, data_dir: Path
) -> dict[str, Callable[[str], tuple[list, BuildReport]]]:
    def from_standard(dataset: str, standard_key: str, registry_source: str | None):
        standard_file = standard_dir / f"{standard_key}.standard.json"
        standard = _load_standard(standard_file)
        # Projection policy comes from DatasetConfig
        # (projection_excluded_category_ids; infra inherits shougang's). The
        # legacy registry is NOT read at build time — audit/parity only.
        categories, report = build_from_standard(
            standard,
            dataset=dataset,
            registry_source=registry_source,
        )
        # corpus.json ``source`` points at the IMMEDIATE input (the canonical
        # standard file); the raw workbook origin is one hop inside that file.
        report.source = _repo_relative(standard_file)
        return categories, report

    return {
        "finance": lambda dataset: from_standard(dataset, "finance", None),
        "shougang": lambda dataset: from_standard(dataset, "shougang", None),
        "infra": lambda dataset: from_standard(dataset, "shougang", "shougang"),
        "pers_info": lambda dataset: build_pers_info_corpus(
            _load_records(data_dir, dataset), dataset
        ),
    }


def _annotate_coverage(
    dataset: str,
    categories: list,
    report: BuildReport,
    data_dir: Path,
) -> None:
    """Read-only registry vs resolver-ID coverage; mutates the report only."""
    records_path = data_dir / "processed" / dataset / "all.json"
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
    report.dataset_id_coverage = {
        "available": True,
        **compute_dataset_id_coverage(dataset, categories, records, config, code_map),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "cfg" / "task")
    parser.add_argument(
        "--standard-dir", type=Path, default=PROJECT_ROOT / "data" / "standards"
    )
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
    builders = _builders(args.standard_dir, data_dir)

    # 1. build everything and compute coverage (no writes yet)
    pending: list[tuple[Path, Any]] = []
    for dataset in args.datasets:
        categories, report = builders[dataset](dataset)
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

    print("wrote:", ", ".join(_display_path(path) for path, _ in pending))
    return 0


def _repo_relative(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
