#!/usr/bin/env python3
"""Inspect and export classification catalogs from clsData-style datasets."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract unique classification paths and label statistics."
    )
    parser.add_argument("--input", required=True, help="Path to a JSON split file.")
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to write catalog metadata as JSON.",
    )
    parser.add_argument(
        "--output-txt",
        default=None,
        help="Optional path to write one leaf path per line.",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=1,
        help="Only export labels with at least this many examples.",
    )
    parser.add_argument(
        "--sort-by",
        choices=["path", "count"],
        default="path",
        help="Sort exported catalog lexicographically or by descending count.",
    )
    parser.add_argument(
        "--path-sep",
        default=" > ",
        help="Separator used in text representation of a four-level path.",
    )
    return parser.parse_args()


def load_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list: {path}")
    return data


def label_tuple(record: dict[str, Any]) -> tuple[str, str, str, str]:
    cls = record.get("classification")
    if not isinstance(cls, dict):
        raise ValueError(f"Missing classification object in record: {record.get('id')}")
    return tuple(str(cls.get(f"level_{i}", "")).strip() for i in range(1, 5))  # type: ignore[return-value]


def path_to_text(path: tuple[str, str, str, str], sep: str) -> str:
    return sep.join(part for part in path if part)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    records = load_json(input_path)

    counts = Counter(label_tuple(record) for record in records)
    items = [(path, count) for path, count in counts.items() if count >= args.min_count]

    if args.sort_by == "count":
        items.sort(key=lambda x: (-x[1], x[0]))
    else:
        items.sort(key=lambda x: x[0])

    by_level: dict[str, Counter[str]] = {
        f"level_{i}": Counter(path[i - 1] for path in counts for _ in range(counts[path]))
        for i in range(1, 5)
    }

    tree: dict[str, Any] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for path, _count in items:
        l1, l2, l3, l4 = path
        tree[l1][l2][l3].append(l4)

    catalog = [
        {
            "path": path_to_text(path, args.path_sep),
            "levels": {
                "level_1": path[0],
                "level_2": path[1],
                "level_3": path[2],
                "level_4": path[3],
            },
            "count": count,
        }
        for path, count in items
    ]

    print(f"input: {input_path}")
    print(f"records: {len(records)}")
    print(f"leaf_categories: {len(counts)}")
    for level_name, counter in by_level.items():
        print(f"{level_name}: {len(counter)}")
        for label, count in counter.most_common(20):
            print(f"  {count}\t{label}")

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "source": str(input_path),
            "num_records": len(records),
            "num_leaf_categories": len(counts),
            "path_separator": args.path_sep,
            "catalog": catalog,
        }
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"wrote: {output_path}")

    if args.output_txt:
        output_path = Path(args.output_txt)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            "\n".join(item["path"] for item in catalog) + "\n",
            encoding="utf-8",
        )
        print(f"wrote: {output_path}")


if __name__ == "__main__":
    main()
