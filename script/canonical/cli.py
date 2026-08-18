"""Build canonical corpus JSON + registry JSON for the four datasets.

Usage:
    python -m script.canonical.cli [--data-dir data] [--out-dir cfg/task] [--overwrite]

Writes per dataset:
    <out-dir>/corpus/<dataset>.corpus.json     canonical corpus (CorpusCategory)
    <out-dir>/registry/<dataset>.registry.json LeafRegistry.from_path-compatible

Sources:
- shougang / infra: data/knowledge/standards_map/guanji_dict.json
- finance:          data/knowledge/standards_map/financial_standards_dict.json
- pers_info:        data/<dataset>/all.json (dataset universe, no standard exists)

The script never modifies data/ or src/ and never repairs labels.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agent.task import BUILTIN_DATASET_CONFIGS  # noqa: E402
from agent.task.canonical_corpus import (  # noqa: E402
    BuildReport,
    build_finance_corpus,
    build_pers_info_corpus,
    build_shougang_corpus,
    compute_dataset_id_coverage,
    corpus_to_mapping,
    registry_to_mapping,
)
from agent.task.identity import code_leaf_map  # noqa: E402
from agent.task.resolver import ClassificationTargetResolver  # noqa: E402


def _write_json(path: Path, data: Any, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"output exists: {path} (pass --overwrite)")
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "cfg" / "task")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    data_dir: Path = args.data_dir
    out_dir: Path = args.out_dir
    standards = data_dir / "knowledge" / "standards_map"

    # shougang + infra (shared guanji universe)
    guanji_path = standards / "guanji_dict.json"
    shougang_categories, shougang_report = build_shougang_corpus(guanji_path, "shougang")
    guanji_code_map = code_leaf_map(shougang_categories)
    _write_bundle("shougang", shougang_categories, shougang_report, out_dir, args.overwrite)
    _write_bundle("infra", shougang_categories, shougang_report, out_dir, args.overwrite)

    # finance
    fs_path = standards / "financial_standards_dict.json"
    finance_categories, finance_report = build_finance_corpus(fs_path, "finance")
    _write_bundle("finance", finance_categories, finance_report, out_dir, args.overwrite)

    # pers_info (dataset universe; needs local data/)
    pers_records = _load_records(data_dir, "pers_info")
    pers_categories, pers_report = build_pers_info_corpus(pers_records, "pers_info")
    _write_bundle("pers_info", pers_categories, pers_report, out_dir, args.overwrite)

    # dataset id coverage diagnostics (registry vs resolver IDs, no mutation)
    _annotate_coverage(
        "shougang", shougang_categories, shougang_report, data_dir, out_dir, args.overwrite
    )
    _annotate_coverage(
        "infra", shougang_categories, shougang_report, data_dir, out_dir, args.overwrite
    )
    _annotate_coverage(
        "finance", finance_categories, finance_report, data_dir, out_dir, args.overwrite
    )
    _annotate_coverage(
        "pers_info", pers_categories, pers_report, data_dir, out_dir, args.overwrite
    )

    print("wrote:", sorted(str(p.relative_to(PROJECT_ROOT)) for p in out_dir.rglob("*.json")
                           if "corpus" in p.parts or "registry" in p.parts))
    return 0


def _write_bundle(
    dataset: str,
    categories: list,
    report: BuildReport,
    out_dir: Path,
    overwrite: bool,
) -> None:
    corpus_path = out_dir / "corpus" / f"{dataset}.corpus.json"
    registry_path = out_dir / "registry" / f"{dataset}.registry.json"
    _write_json(
        corpus_path,
        corpus_to_mapping(dataset, report.source, report.id_strategy, categories, report),
        overwrite,
    )
    _write_json(registry_path, registry_to_mapping(categories), overwrite)


def _annotate_coverage(
    dataset: str,
    categories: list,
    report: BuildReport,
    data_dir: Path,
    out_dir: Path,
    overwrite: bool,
) -> None:
    records_path = data_dir / dataset / "all.json"
    if not records_path.is_file():
        report.dataset_id_coverage = {"available": False}
        _write_json(out_dir / "corpus" / f"{dataset}.corpus.json",
                    corpus_to_mapping(dataset, report.source, report.id_strategy,
                                      categories, report), overwrite)
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
    _write_json(out_dir / "corpus" / f"{dataset}.corpus.json",
                corpus_to_mapping(dataset, report.source, report.id_strategy,
                                  categories, report), overwrite)


if __name__ == "__main__":
    raise SystemExit(main())
