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


def _joint_labels(record: dict[str, Any]) -> tuple[str | None, str | None]:
    target = record.get("target")
    category = target.get("category_id") if isinstance(target, dict) else None
    level = record.get("data_level")
    return (
        category.strip() if isinstance(category, str) and category.strip() else None,
        level.strip() if isinstance(level, str) and level.strip() else None,
    )


def _train_coverage_gaps(
    splits: tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]
) -> tuple[list[str], list[str]]:
    train_categories: set[str] = set()
    train_levels: set[str] = set()
    later_categories: set[str] = set()
    later_levels: set[str] = set()
    for split_index, split in enumerate(splits):
        for record in split:
            category, level = _joint_labels(record)
            if split_index == 0:
                if category:
                    train_categories.add(category)
                if level:
                    train_levels.add(level)
            else:
                if category:
                    later_categories.add(category)
                if level:
                    later_levels.add(level)
    return (
        sorted(later_categories - train_categories),
        sorted(later_levels - train_levels),
    )


def ensure_train_coverage(
    splits: tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]
) -> tuple[
    tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]],
    dict[str, Any],
]:
    """Deterministically swap records so train covers every leaf and level.

    Counts remain unchanged. A donor may leave train only when both of its
    labels still have another train occurrence; otherwise the requested
    no-gap policy is impossible at the configured split sizes and fails.
    """

    result = [list(split) for split in splits]
    swaps: list[dict[str, str]] = []

    def counts() -> tuple[Counter[str], Counter[str]]:
        categories: Counter[str] = Counter()
        levels: Counter[str] = Counter()
        for record in result[0]:
            category, level = _joint_labels(record)
            if category:
                categories[category] += 1
            if level:
                levels[level] += 1
        return categories, levels

    def gaps() -> tuple[list[str], list[str]]:
        train_categories, train_levels = counts()
        later_categories: set[str] = set()
        later_levels: set[str] = set()
        for split in result[1:]:
            for record in split:
                category, level = _joint_labels(record)
                if category:
                    later_categories.add(category)
                if level:
                    later_levels.add(level)
        return (
            sorted(later_categories - set(train_categories)),
            sorted(later_levels - set(train_levels)),
        )

    for _ in range(sum(len(split) for split in result) + 1):
        category_gaps, level_gaps = gaps()
        if not category_gaps and not level_gaps:
            break
        kind = "category" if category_gaps else "level"
        missing = (category_gaps or level_gaps)[0]
        source_index = None
        incoming = None
        for later_index in (1, 2):
            candidates = sorted(result[later_index], key=lambda row: str(row.get("id", "")))
            for record in candidates:
                category, level = _joint_labels(record)
                if (category if kind == "category" else level) == missing:
                    source_index, incoming = later_index, record
                    break
            if incoming is not None:
                break
        if source_index is None or incoming is None:
            raise ValueError(f"cannot locate record for missing train {kind} label")
        category_counts, level_counts = counts()
        outgoing = None
        for record in sorted(result[0], key=lambda row: str(row.get("id", ""))):
            category, level = _joint_labels(record)
            if category and category_counts[category] <= 1:
                continue
            if level and level_counts[level] <= 1:
                continue
            outgoing = record
            break
        if outgoing is None:
            raise ValueError(
                "cannot repair train category/data_level gaps without creating another gap"
            )
        result[0].remove(outgoing)
        result[source_index].remove(incoming)
        result[0].append(incoming)
        result[source_index].append(outgoing)
        swaps.append(
            {
                "split": SPLIT_NAMES[source_index],
                "incoming_id": str(incoming.get("id", "")),
                "outgoing_id": str(outgoing.get("id", "")),
                "reason": f"missing_{kind}",
            }
        )
    category_gaps, level_gaps = gaps()
    if category_gaps or level_gaps:
        raise ValueError(
            f"train coverage repair incomplete: categories={category_gaps}, levels={level_gaps}"
        )
    for split in result:
        split.sort(key=lambda row: str(row.get("id", "")))
    return (
        (result[0], result[1], result[2]),
        {
            "policy": "train covers every category_id and data_level",
            "swaps": len(swaps),
            "details": swaps,
            "remaining_category_gaps": category_gaps,
            "remaining_level_gaps": level_gaps,
        },
    )


def prepare_split(
    dataset: str,
    *,
    canonical_dir: str | Path,
    split_type: str = "random",
    group_key: str | None = None,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
    require_train_coverage: bool = False,
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
    coverage_report: dict[str, Any] | None = None
    if split_type == "random":
        raw_splits = stratified_split(pool, ratios, seed)
        if require_train_coverage:
            splits, coverage_report = ensure_train_coverage(raw_splits)
            coverage_report["enforced"] = True
        else:
            splits = raw_splits
            category_gaps, level_gaps = _train_coverage_gaps(raw_splits)
            coverage_report = {
                "policy": "train coverage recorded but not enforced",
                "enforced": False,
                "swaps": 0,
                "details": [],
                "remaining_category_gaps": category_gaps,
                "remaining_level_gaps": level_gaps,
            }
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
            "algorithm_version": (
                SPLIT_ALGORITHM_VERSION
                if split_type == "group"
                else (
                    f"{SPLIT_ALGORITHM_VERSION}+train-coverage-v1"
                    if require_train_coverage
                    else SPLIT_ALGORITHM_VERSION
                )
            ),
        }
    )
    if split_type == "group":
        report["group_key"] = group_key
    else:
        report["train_coverage_gate"] = coverage_report
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
    parser.add_argument(
        "--require-train-coverage",
        action="store_true",
        help="Fail unless train covers every category_id and data_level",
    )
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
            require_train_coverage=args.require_train_coverage,
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
