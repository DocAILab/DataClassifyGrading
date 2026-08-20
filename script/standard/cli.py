"""Phase 1 canonical standard CLI: build standards + alignment artifacts.

Usage:
    python -m script.standard.cli [--overwrite]

Reads the ORIGINAL standard workbooks (data/raw) and the canonical dataset
records (data/canonical), then writes:

    data/standards/finance.standard.json
    data/standards/shougang.standard.json
    artifacts/generated/provenance/finance_standard_alignment.json
    artifacts/generated/provenance/shougang_standard_alignment.json
    artifacts/generated/provenance/infra_standard_alignment.json
    artifacts/generated/provenance/standard_build_summary.json

Fail-fast: every dataset is read + built + aligned before anything is
written; all outputs are written exactly once. Deterministic: JSON is written
with sort_keys and categories/entries are pre-sorted; no timestamps or
machine-local paths. Raw workbooks are never training dependencies.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agent.standards.align import align_dataset_to_standard, load_canonical_records
from agent.standards.build import (
    build_finance_standard,
    build_shougang_standard,
    resolve_standard_dataset,
)
from agent.standards.contracts import CanonicalStandard
from agent.standards.sources import (
    read_finance_standard_guide,
    read_guanji_catalog,
)

DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_CANONICAL_DIR = PROJECT_ROOT / "data" / "canonical"
DEFAULT_STANDARD_DIR = PROJECT_ROOT / "data" / "standards"
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "generated" / "provenance"

FINANCE_XLSX = "金融行业数据安全分类分级标准指南.xlsx"
SHOUGANG_XLSX = "关基-数据分类分级目录.xlsx"


def _repo_relative(path: str | Path) -> str:
    try:
        return Path(path).resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return Path(path).as_posix()


def _write_json(payload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _registry_ids(dataset: str) -> set[str]:
    from agent.task import LeafRegistry

    path = PROJECT_ROOT / "cfg" / "task" / "registry" / f"{dataset}.registry.json"
    return set(LeafRegistry.from_path(path).ids)


def _legacy_information_loss(
    finance_standard: CanonicalStandard,
    shougang_standard: CanonicalStandard,
) -> dict[str, object]:
    """Quantify what the legacy standards_map digests dropped vs the raw
    standard canonical build (path depth, grading, missing codes)."""

    import json as _json
    import re as _re

    finance_out: dict[str, object] = {}
    finance_ids = {c.category_id for c in finance_standard.categories}
    finance_reg_ids = _registry_ids("finance")
    finance_out["standard_vs_registry_ids"] = {
        "standard_ids": len(finance_ids),
        "registry_ids": len(finance_reg_ids),
        "missing_from_registry": len(finance_ids - finance_reg_ids),
    }
    # legacy dict: how many categories lost a real path level
    legacy = _json.load(
        (
            PROJECT_ROOT / "data" / "knowledge" / "standards_map"
            / "financial_standards_dict.json"
        ).open(encoding="utf-8")
    )
    depth_lost = 0
    depth_kept = 0
    for category in finance_standard.categories:
        segments = 3  # legacy dict identity was L1-L2-leaf
        if len(category.path) > segments:
            depth_lost += 1
        else:
            depth_kept += 1
    finance_out["legacy_dict_path_compression"] = {
        "categories_with_path_deeper_than_legacy_L1_L2_leaf": depth_lost,
        "categories_at_legacy_depth": depth_kept,
        "note": "legacy financial_standards_dict stored L1-L2-leaf identity "
        "strings; the real standard has 三级子类 provenance nodes that the "
        "legacy digest dropped",
    }
    finance_out["legacy_dict_entries"] = len(legacy)
    finance_out["legacy_unparseable_level_values"] = sorted(
        str(v.get("class", "")) for v in legacy.values() if isinstance(v, dict)
        and v.get("class") in ("l级", "3 4级")
    )

    shougang_out: dict[str, object] = {}
    shougang_ids = {c.category_id for c in shougang_standard.categories}
    shougang_reg_ids = _registry_ids("shougang")
    shougang_out["standard_vs_registry_codes"] = {
        "standard_codes": len(shougang_ids),
        "registry_codes": len(shougang_reg_ids),
        "standard_only": sorted(shougang_ids - shougang_reg_ids),
        "registry_only": sorted(shougang_reg_ids - shougang_ids),
        "note": "B3-6 中厚板作业计划 exists in the raw catalog but was dropped "
        "by guanji_dict/registry",
    }
    legacy_g = _json.load(
        (
            PROJECT_ROOT / "data" / "knowledge" / "standards_map" / "guanji_dict.json"
        ).open(encoding="utf-8")
    )
    codes_without_path = 0
    codes_with_level = 0
    for category in shougang_standard.categories:
        if category.path:
            codes_without_path += 1  # path restored where legacy had none
        if category.standard_data_level:
            codes_with_level += 1  # grading restored where legacy had none
    shougang_out["legacy_dict_losses"] = {
        "catalog_categories": len(shougang_standard.categories),
        "legacy_dict_entries": len(legacy_g),
        "with_real_path_restored": codes_without_path,
        "with_grading_restored": codes_with_level,
        "note": "guanji_dict kept only 'name（code）' + description: real "
        "hierarchy path and the 分级 column were dropped (registry path was "
        "[] and no class field existed)",
    }
    return {"finance": finance_out, "shougang": shougang_out}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--canonical-dir", type=Path, default=DEFAULT_CANONICAL_DIR)
    parser.add_argument("--standard-dir", type=Path, default=DEFAULT_STANDARD_DIR)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    raw_dir = Path(args.raw_dir)
    finance_xlsx = raw_dir / FINANCE_XLSX
    shougang_xlsx = raw_dir / SHOUGANG_XLSX
    missing = [p for p in (finance_xlsx, shougang_xlsx) if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "raw standard workbook(s) missing (data/raw is gitignored; restore "
            "them from the data provider before building): "
            + ", ".join(str(p) for p in missing)
        )

    # 1. read + build + align EVERYTHING (pure computation, no writes)
    finance_raw = read_finance_standard_guide(finance_xlsx)
    shougang_raw = read_guanji_catalog(shougang_xlsx)
    finance_standard, finance_report = build_finance_standard(
        finance_raw.entries,
        source_file=_repo_relative(finance_xlsx),
        source_sheet="Table 1",
    )
    shougang_standard, shougang_report = build_shougang_standard(
        shougang_raw.entries,
        source_file=_repo_relative(shougang_xlsx),
        source_sheet="数据分类分级",
    )

    canonical_records = {}
    for dataset in ("finance", "shougang", "infra"):
        canonical_records[dataset] = load_canonical_records(
            args.canonical_dir / dataset / "all.json"
        )

    standard_key = resolve_standard_dataset("infra")  # -> shougang (shared)
    alignments = {
        "finance": align_dataset_to_standard(
            canonical_records["finance"], finance_standard
        ),
        "shougang": align_dataset_to_standard(
            canonical_records["shougang"], shougang_standard
        ),
        "infra": align_dataset_to_standard(
            canonical_records["infra"], shougang_standard
        ),
    }

    loss = _legacy_information_loss(finance_standard, shougang_standard)

    # 2. refuse to overwrite without --overwrite
    outputs = {
        "standards/finance": args.standard_dir / "finance.standard.json",
        "standards/shougang": args.standard_dir / "shougang.standard.json",
        "provenance/finance_alignment": args.artifact_dir / "finance_standard_alignment.json",
        "provenance/shougang_alignment": args.artifact_dir / "shougang_standard_alignment.json",
        "provenance/infra_alignment": args.artifact_dir / "infra_standard_alignment.json",
        "provenance/summary": args.artifact_dir / "standard_build_summary.json",
    }
    if not args.overwrite:
        existing = [str(p) for p in outputs.values() if p.exists()]
        if existing:
            raise FileExistsError(
                "refusing to overwrite existing canonical-standard artifacts: "
                + ", ".join(existing)
                + " (pass --overwrite to regenerate)"
            )

    # 3. build the summary first (fail-fast: nothing is written until every
    #    computed payload is ready)
    summary = {
        "phase": "phase1-canonical-standard",
        "standards": {
            dataset: {
                "standard_source": (
                    resolve_standard_dataset(dataset)
                    if resolve_standard_dataset(dataset)
                    else None
                ),
                "status": (
                    "missing_or_unknown"
                    if resolve_standard_dataset(dataset) is None
                    else "built"
                ),
                "categories": (
                    len(
                        {
                            "finance": finance_standard,
                            "shougang": shougang_standard,
                        }[resolve_standard_dataset(dataset)].categories
                    )
                    if resolve_standard_dataset(dataset) in ("finance", "shougang")
                    else 0
                ),
                "build": (
                    finance_report.to_mapping()
                    if dataset == "finance"
                    else shougang_report.to_mapping()
                    if dataset in ("shougang", "infra")
                    else None
                ),
                "alignment_headline": {
                    "samples_total": alignments[dataset]["sample_counts"].get("total", 0),
                    "resolved": alignments[dataset]["sample_counts"].get("resolved", 0),
                    "matched": alignments[dataset]["sample_counts"].get("matched", 0),
                    "mismatched": alignments[dataset]["sample_counts"].get("mismatched", 0),
                    "standard_missing": alignments[dataset]["sample_counts"].get("standard_missing", 0),
                    "resolved_match_rate": alignments[dataset]["resolved_match_rate"],
                },
            }
            for dataset in ("finance", "shougang", "infra")
        },
        "pers_info": {
            "standard_source": None,
            "status": "missing_or_unknown",
            "note": "no confirmed classification/grading standard; the 18-category "
            "registry remains dataset-derived and is NOT presented as a canonical "
            "standard; no standard_data_level is fabricated",
            "alignment_headline": None,
        },
        "legacy_information_loss": loss,
    }
    # 4. write everything exactly once
    _write_json(finance_standard.to_mapping(), outputs["standards/finance"])
    _write_json(shougang_standard.to_mapping(), outputs["standards/shougang"])
    for key in ("finance", "shougang", "infra"):
        _write_json(alignments[key], outputs[f"provenance/{key}_alignment"])
    _write_json(summary, outputs["provenance/summary"])

    for name, path in outputs.items():
        print(f"wrote: {_repo_relative(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
