#!/usr/bin/env python3
"""Build one-turn SFT datasets for field classification."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


DEFAULT_SYSTEM_PROMPT = (
    "你的任务是给定数据库中的某个字段，以及数据分类目录，输出其分类。"
    "必须只从给定的数据分类目录中选择一个四级分类；不要解释，不要输出推理过程。"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build LLaMA-Factory style one-turn SFT data."
    )
    parser.add_argument("--dataset-root", required=True, help="Path to clsData root.")
    parser.add_argument("--domain", default="shougang", help="Dataset domain folder.")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Splits to build, e.g. train val test.",
    )
    parser.add_argument(
        "--catalog-source",
        default="all",
        help="Split used to build the catalog list. Usually all.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for outputs.")
    parser.add_argument(
        "--format",
        choices=["sharegpt", "alpaca"],
        default="sharegpt",
        help="Output data format.",
    )
    parser.add_argument(
        "--output-format",
        choices=["json", "path", "list"],
        default="json",
        help="Assistant answer format.",
    )
    parser.add_argument(
        "--field-source",
        choices=["field_name", "key"],
        default="field_name",
        help="Which source to use as field_name in the user JSON.",
    )
    parser.add_argument(
        "--field-case",
        choices=["original", "lower"],
        default="original",
        help="Normalize field_name casing.",
    )
    parser.add_argument(
        "--include-empty-labels",
        action="store_true",
        help="Keep records whose label path contains an empty level.",
    )
    parser.add_argument(
        "--include-metadata",
        nargs="*",
        default=[],
        choices=[
            "field_description",
            "table_name",
            "table_description",
            "database_name",
            "field_type",
        ],
        help="Optional metadata keys to add into the user JSON.",
    )
    parser.add_argument(
        "--catalog-shuffle",
        action="store_true",
        help="Shuffle catalog order per example.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when shuffling catalog order.",
    )
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt text.",
    )
    parser.add_argument(
        "--path-sep",
        default=" > ",
        help="Separator used for path string outputs and catalog entries.",
    )
    parser.add_argument(
        "--write-dataset-info",
        action="store_true",
        help="Write a LLaMA-Factory dataset_info.json beside generated data.",
    )
    return parser.parse_args()


def load_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list: {path}")
    return data


def get_label(record: dict[str, Any]) -> dict[str, str]:
    cls = record.get("classification")
    if not isinstance(cls, dict):
        raise ValueError(f"Missing classification object in record: {record.get('id')}")
    return {f"level_{i}": str(cls.get(f"level_{i}", "")).strip() for i in range(1, 5)}


def label_to_tuple(label: dict[str, str]) -> tuple[str, str, str, str]:
    return tuple(label[f"level_{i}"] for i in range(1, 5))  # type: ignore[return-value]


def path_to_text(path: tuple[str, str, str, str], sep: str) -> str:
    return sep.join(part for part in path if part)


def build_catalog(records: list[dict[str, Any]], include_empty: bool, sep: str) -> list[str]:
    paths: set[tuple[str, str, str, str]] = set()
    for record in records:
        path = label_to_tuple(get_label(record))
        if not include_empty and any(not part for part in path):
            continue
        paths.add(path)
    return [path_to_text(path, sep) for path in sorted(paths)]


def field_value(record: dict[str, Any], source: str, case: str) -> str:
    metadata = record.get("metadata") or {}
    value = metadata.get("field_name") if source == "field_name" else record.get("key")
    value = str(value or "").strip()
    if case == "lower":
        value = value.lower()
    return value


def assistant_answer(label: dict[str, str], output_format: str, sep: str) -> str:
    path = label_to_tuple(label)
    if output_format == "path":
        return path_to_text(path, sep)
    if output_format == "list":
        return json.dumps(list(path), ensure_ascii=False)
    return json.dumps(label, ensure_ascii=False, separators=(",", ":"))


def make_user_payload(
    record: dict[str, Any],
    catalog: list[str],
    args: argparse.Namespace,
    rng: random.Random,
) -> str:
    metadata = record.get("metadata") or {}
    example_catalog = list(catalog)
    if args.catalog_shuffle:
        rng.shuffle(example_catalog)

    payload: dict[str, Any] = {
        "field_name": field_value(record, args.field_source, args.field_case),
        "catalog": example_catalog,
    }
    for key in args.include_metadata:
        payload[key] = str(metadata.get(key, "")).strip()
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def to_sharegpt(system: str, user: str, answer: str) -> dict[str, Any]:
    return {
        "system": system,
        "conversations": [
            {"from": "human", "value": user},
            {"from": "gpt", "value": answer},
        ],
    }


def to_alpaca(system: str, user: str, answer: str) -> dict[str, Any]:
    return {
        "instruction": system,
        "input": user,
        "output": answer,
    }


def build_split(
    records: list[dict[str, Any]],
    catalog: list[str],
    args: argparse.Namespace,
    split: str,
) -> list[dict[str, Any]]:
    rng = random.Random(args.seed + sum(ord(ch) for ch in split))
    output: list[dict[str, Any]] = []
    skipped = 0
    for record in records:
        label = get_label(record)
        if not args.include_empty_labels and any(not v for v in label.values()):
            skipped += 1
            continue
        user = make_user_payload(record, catalog, args, rng)
        answer = assistant_answer(label, args.output_format, args.path_sep)
        if args.format == "sharegpt":
            item = to_sharegpt(args.system_prompt, user, answer)
        else:
            item = to_alpaca(args.system_prompt, user, answer)
        item["id"] = record.get("id", "")
        output.append(item)
    print(f"{split}: wrote {len(output)} examples, skipped {skipped}")
    return output


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_dataset_info(output_dir: Path, domain: str, splits: list[str], fmt: str) -> None:
    info: dict[str, Any] = {}
    for split in splits:
        name = f"{domain}_sft_{split}"
        file_name = f"{domain}_sft_{split}.json"
        if fmt == "sharegpt":
            info[name] = {
                "file_name": file_name,
                "formatting": "sharegpt",
                "columns": {
                    "messages": "conversations",
                    "system": "system",
                },
                "tags": {
                    "role_tag": "from",
                    "content_tag": "value",
                    "user_tag": "human",
                    "assistant_tag": "gpt",
                },
            }
        else:
            info[name] = {
                "file_name": file_name,
                "columns": {
                    "prompt": "instruction",
                    "query": "input",
                    "response": "output",
                },
            }
    write_json(output_dir / "dataset_info.json", info)


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    domain_dir = dataset_root / args.domain
    output_dir = Path(args.output_dir)

    catalog_records = load_json(domain_dir / f"{args.catalog_source}.json")
    catalog = build_catalog(catalog_records, args.include_empty_labels, args.path_sep)
    if not catalog:
        raise ValueError("Catalog is empty. Check --catalog-source and labels.")

    write_json(output_dir / f"{args.domain}_catalog.json", catalog)
    print(f"catalog: {len(catalog)} leaf paths")

    built_splits: list[str] = []
    for split in args.splits:
        split_path = domain_dir / f"{split}.json"
        records = load_json(split_path)
        examples = build_split(records, catalog, args, split)
        write_json(output_dir / f"{args.domain}_sft_{split}.json", examples)
        built_splits.append(split)

    if args.write_dataset_info:
        write_dataset_info(output_dir, args.domain, built_splits, args.format)
        print(f"wrote: {output_dir / 'dataset_info.json'}")


if __name__ == "__main__":
    main()
