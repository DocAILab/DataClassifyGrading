"""Deterministic, resolution-aware splitting of the canonical contract layer.

Unlike the processed-layer splitter (``script.preprocessing.split``), this
operates AFTER canonical resolution so that exclusion decisions are explicit
per record and written back into ``all.json``:

- ``resolution_status != "resolved"``  ->  ``split=null``,
  ``split_exclusion_reason="resolution_status:<status>"`` (they must never
  enter training);
- group splits skip records whose group key is empty ->
  ``split_exclusion_reason="empty_group_key:<key>"``;
- everything else is stratified/group-split deterministically after a full
  sort by stable id (row-order insensitivity), with seed/ratios/order
  recorded in ``split_report.json``.

Outputs per dataset:
    <canonical-dir>/<dataset>/all.json          (split fields filled in)
    <canonical-dir>/<dataset>/train|val|test.json   (record views, compat)
    <canonical-dir>/<dataset>/split_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from script.preprocessing.split import (
    SPLIT_ALGORITHM_VERSION,
    build_split_report,
    canonical_record_order,
    group_split,
    stratified_split,
)

SPLIT_NAMES = ("train", "val", "test")


def _load(path: Path) -> tuple[list[dict[str, Any]], str]:
    if not path.is_file():
        raise FileNotFoundError(f"canonical dataset not found: {path}")
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{path} must contain a non-empty JSON list")
    return value, path.read_text(encoding="utf-8")


def _group_value(record: dict[str, Any], group_key: str) -> str:
    value: Any = record
    for part in group_key.split("."):
        if not isinstance(value, dict) or part not in value:
            return ""
        value = value[part]
    return "" if value is None else str(value).strip()


def prepare_split(
    dataset: str,
    *,
    canonical_dir: str | Path,
    split_type: str = "random",
    group_key: str | None = None,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """Compute splits for one canonical dataset. Pure computation.

    Returns ``(report, enriched_records, views)`` where ``views`` maps
    train/val/test to record lists. Nothing is written here.
    """
    if split_type not in {"random", "group"}:
        raise ValueError("split_type must be 'random' or 'group'")
    if split_type == "group" and not group_key:
        raise ValueError("group_key is required for a group split")
    if any(ratio < 0 or ratio > 1 for ratio in ratios) or abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("ratios must be within [0, 1] and sum to 1")
    if ratios[0] == 0:
        raise ValueError("train ratio must be greater than zero")

    root = Path(canonical_dir) / dataset
    records, _ = _load(root / "all.json")

    excluded: list[dict[str, Any]] = []
    pool: list[dict[str, Any]] = []
    for record in records:
        status = str(record.get("resolution_status", "") or "").strip()
        if status != "resolved":
            record["split"] = None
            record["split_exclusion_reason"] = f"resolution_status:{status or 'missing'}"
            excluded.append(
                {"id": str(record.get("id", "")), "reason": record["split_exclusion_reason"]}
            )
            continue
        if split_type == "group" and not _group_value(record, group_key):
            record["split"] = None
            record["split_exclusion_reason"] = f"empty_group_key:{group_key}"
            excluded.append(
                {"id": str(record.get("id", "")), "reason": record["split_exclusion_reason"]}
            )
            continue
        record["split"] = None
        record["split_exclusion_reason"] = None
        pool.append(record)

    pool = canonical_record_order(pool)
    if split_type == "random":
        splits = stratified_split(pool, ratios, seed)
    else:
        splits = group_split(pool, group_key, ratios, seed)

    views: dict[str, list[dict[str, Any]]] = {}
    for name, split_records in zip(SPLIT_NAMES, splits):
        for record in split_records:
            record["split"] = name
        views[name] = split_records

    report = build_split_report(tuple(views[name] for name in SPLIT_NAMES))
    reason_counts = Counter(item["reason"] for item in excluded)
    report.update(
        {
            "dataset": dataset,
            "input_size": len(records),
            "included_size": len(pool),
            "excluded": {
                "count": len(excluded),
                "by_reason": dict(sorted(reason_counts.items())),
                "ids": [item["id"] for item in excluded],
            },
            "seed": seed,
            "split_type": split_type,
            "ratios": {
                "train": ratios[0],
                "val": ratios[1],
                "test": ratios[2],
            },
            "order_rule": "id-ascending",
            "algorithm_version": SPLIT_ALGORITHM_VERSION,
        }
    )
    if split_type == "group":
        report["group_key"] = group_key
    return report, records, views


def write_split(
    dataset: str,
    *,
    canonical_dir: str | Path,
    report: dict[str, Any],
    records: list[dict[str, Any]],
    views: dict[str, list[dict[str, Any]]],
    overwrite: bool = False,
) -> None:
    """Atomically persist all.json (write-back), split views and report."""
    root = Path(canonical_dir) / dataset
    outputs = {
        "all.json": records,
        **{f"{name}.json": views[name] for name in SPLIT_NAMES},
        "split_report.json": report,
    }
    existing = [name for name in outputs if (root / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite existing output in {root}: {sorted(existing)} "
            "(pass --overwrite)"
        )
    root.mkdir(parents=True, exist_ok=True)
    staged: list[tuple[Path, Path]] = []
    with tempfile.TemporaryDirectory(prefix=f".{dataset}.split.", dir=root.parent) as tmp:
        for name, payload in outputs.items():
            source = Path(tmp) / name
            with source.open("w", encoding="utf-8", newline="\n") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            staged.append((source, root / name))
        for source, destination in staged:
            destination.parent.mkdir(parents=True, exist_ok=True)
            source.replace(destination)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--canonical-dir", required=True,
                        help="Directory containing <dataset>/all.json")
    parser.add_argument("--dataset", action="append", dest="datasets", required=True,
                        help="Dataset to split; repeatable")
    parser.add_argument("--split-type", choices=("random", "group"), default="random")
    parser.add_argument("--group-key",
                        help="Dotted path used as group key, e.g. metadata.table_name")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.split_type == "group" and not args.group_key:
        raise SystemExit("--group-key is required when --split-type=group")
    ratios = (args.train_ratio, args.val_ratio, args.test_ratio)
    summaries = {}
    for dataset in args.datasets:
        report, records, views = prepare_split(
            dataset,
            canonical_dir=args.canonical_dir,
            split_type=args.split_type,
            group_key=args.group_key,
            ratios=ratios,
            seed=args.seed,
        )
        write_split(
            dataset,
            canonical_dir=args.canonical_dir,
            report=report,
            records=records,
            views=views,
            overwrite=args.overwrite,
        )
        summaries[dataset] = {
            "sizes": report["sizes"],
            "excluded": report["excluded"]["count"],
        }
    print(json.dumps({"status": "ok", "datasets": summaries}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
