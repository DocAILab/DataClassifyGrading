#!/usr/bin/env python3
"""Normalize clsData records for auditing and downstream SFT building."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess clsData domain splits.")
    parser.add_argument("--dataset-root", required=True, help="Path to clsData root.")
    parser.add_argument("--domain", default="shougang", help="Domain folder name.")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test", "all"],
        help="Splits to normalize.",
    )
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument(
        "--path-sep",
        default=" > ",
        help="Separator used for classification_path.",
    )
    parser.add_argument(
        "--fail-on-empty-label",
        action="store_true",
        help="Fail if any classification level is empty.",
    )
    return parser.parse_args()


def load_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list: {path}")
    return data


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def get_label(record: dict[str, Any]) -> dict[str, str]:
    cls = record.get("classification")
    if not isinstance(cls, dict):
        raise ValueError(f"Missing classification object in record: {record.get('id')}")
    return {f"level_{i}": str(cls.get(f"level_{i}", "")).strip() for i in range(1, 5)}


def normalize_record(record: dict[str, Any], sep: str) -> dict[str, Any]:
    metadata = record.get("metadata") or {}
    label = get_label(record)
    levels = [label[f"level_{i}"] for i in range(1, 5)]
    return {
        "id": record.get("id", ""),
        "key": record.get("key", ""),
        "field_name": str(metadata.get("field_name", "")).strip(),
        "field_description": str(metadata.get("field_description", "")).strip(),
        "field_type": str(metadata.get("field_type", "")).strip(),
        "table_name": str(metadata.get("table_name", "")).strip(),
        "table_description": str(metadata.get("table_description", "")).strip(),
        "database_name": str(metadata.get("database_name", "")).strip(),
        "database_description": str(metadata.get("database_description", "")).strip(),
        "classification": label,
        "classification_path": sep.join(level for level in levels if level),
        "data_level": record.get("data_level", ""),
        "label_status": record.get("label_status", ""),
    }


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    domain_dir = dataset_root / args.domain
    output_dir = Path(args.output_dir)

    all_paths: Counter[str] = Counter()
    summary: dict[str, Any] = {
        "dataset_root": str(dataset_root),
        "domain": args.domain,
        "splits": {},
    }

    for split in args.splits:
        input_path = domain_dir / f"{split}.json"
        records = load_json(input_path)
        normalized = [normalize_record(record, args.path_sep) for record in records]

        empty_label_ids = [
            item["id"]
            for item in normalized
            if any(not item["classification"][f"level_{i}"] for i in range(1, 5))
        ]
        if args.fail_on_empty_label and empty_label_ids:
            raise ValueError(
                f"{split} has {len(empty_label_ids)} records with empty label levels"
            )

        path_counts = Counter(item["classification_path"] for item in normalized)
        all_paths.update(path_counts)
        summary["splits"][split] = {
            "input": str(input_path),
            "output": str(output_dir / f"{args.domain}_{split}_normalized.json"),
            "records": len(normalized),
            "leaf_categories": len(path_counts),
            "empty_label_records": len(empty_label_ids),
        }
        write_json(output_dir / f"{args.domain}_{split}_normalized.json", normalized)
        print(
            f"{split}: records={len(normalized)} "
            f"leaf_categories={len(path_counts)} empty_label_records={len(empty_label_ids)}"
        )

    catalog = [
        {"path": path, "count": count}
        for path, count in sorted(all_paths.items(), key=lambda item: item[0])
    ]
    summary["catalog"] = {
        "leaf_categories": len(catalog),
        "output": str(output_dir / f"{args.domain}_catalog_from_preprocess.json"),
    }
    write_json(output_dir / f"{args.domain}_catalog_from_preprocess.json", catalog)
    write_json(output_dir / "preprocess_summary.json", summary)
    print(f"catalog: {len(catalog)}")
    print(f"wrote: {output_dir}")


if __name__ == "__main__":
    main()
