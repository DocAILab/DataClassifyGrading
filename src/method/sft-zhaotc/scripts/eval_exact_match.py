#!/usr/bin/env python3
"""Exact-match evaluation for one-turn classification outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate exact match only.")
    parser.add_argument(
        "--reference",
        required=True,
        help="Reference SFT file (sharegpt or alpaca) or a JSONL with gold labels.",
    )
    parser.add_argument(
        "--prediction",
        required=True,
        help="Prediction file: txt (one line per sample), JSONL, or JSON list.",
    )
    parser.add_argument(
        "--reference-format",
        choices=["auto", "sharegpt", "alpaca", "jsonl", "json"],
        default="auto",
        help="Reference file format.",
    )
    parser.add_argument(
        "--prediction-format",
        choices=["auto", "txt", "jsonl", "json"],
        default="auto",
        help="Prediction file format.",
    )
    parser.add_argument(
        "--prediction-field",
        default="prediction",
        help="Field name used when prediction file is JSON/JSONL.",
    )
    parser.add_argument(
        "--gold-field",
        default="gold",
        help="Field name used when reference file is JSONL with gold labels.",
    )
    parser.add_argument(
        "--output-report",
        default=None,
        help="Optional path to write a JSON report.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def detect_reference_format(path: Path) -> str:
    if path.suffix == ".jsonl":
        return "jsonl"
    if path.suffix == ".json":
        data = load_json(path)
        if isinstance(data, list) and data:
            first = data[0]
            if isinstance(first, dict) and "conversations" in first:
                return "sharegpt"
            if isinstance(first, dict) and {"instruction", "input", "output"} <= set(first):
                return "alpaca"
            return "json"
    return "auto"


def extract_gold_from_sharegpt(item: dict[str, Any]) -> str:
    conversations = item.get("conversations")
    if not isinstance(conversations, list):
        raise ValueError("sharegpt record missing conversations")
    for message in reversed(conversations):
        if isinstance(message, dict) and message.get("from") == "gpt":
            return str(message.get("value", ""))
    raise ValueError("sharegpt record missing gpt answer")


def extract_gold_from_alpaca(item: dict[str, Any]) -> str:
    return str(item.get("output", ""))


def normalize_text(text: str) -> str:
    return text.strip()


def parse_label(text: str) -> Any:
    text = text.strip()
    if not text:
        raise ValueError("empty prediction")
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return text
    if isinstance(obj, dict):
        if {"level_1", "level_2", "level_3", "level_4"} <= set(obj):
            return {f"level_{i}": str(obj.get(f"level_{i}", "")).strip() for i in range(1, 5)}
        return obj
    if isinstance(obj, list):
        return [str(x).strip() for x in obj]
    return obj


def canonicalize(label: Any) -> str:
    if isinstance(label, dict):
        if {"level_1", "level_2", "level_3", "level_4"} <= set(label):
            obj = {f"level_{i}": str(label.get(f"level_{i}", "")).strip() for i in range(1, 5)}
            return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        return json.dumps(label, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    if isinstance(label, list):
        return json.dumps([str(x).strip() for x in label], ensure_ascii=False, separators=(",", ":"))
    return str(label).strip()


def load_reference(path: Path, fmt: str) -> list[str]:
    if fmt == "auto":
        fmt = detect_reference_format(path)

    if fmt == "sharegpt":
        data = load_json(path)
        if not isinstance(data, list):
            raise ValueError("sharegpt reference must be a JSON list")
        return [canonicalize(parse_label(extract_gold_from_sharegpt(item))) for item in data]
    if fmt == "alpaca":
        data = load_json(path)
        if not isinstance(data, list):
            raise ValueError("alpaca reference must be a JSON list")
        return [canonicalize(parse_label(extract_gold_from_alpaca(item))) for item in data]
    if fmt == "jsonl":
        rows = load_jsonl(path)
        return [canonicalize(parse_label(str(row["gold"]))) for row in rows]
    if fmt == "json":
        data = load_json(path)
        if not isinstance(data, list):
            raise ValueError("reference json must be a JSON list")
        return [canonicalize(parse_label(str(item))) for item in data]
    raise ValueError(f"Unsupported reference format: {fmt}")


def load_predictions(path: Path, fmt: str, field: str) -> list[str]:
    if fmt == "auto":
        if path.suffix == ".txt":
            fmt = "txt"
        elif path.suffix == ".jsonl":
            fmt = "jsonl"
        else:
            fmt = "json"

    if fmt == "txt":
        with path.open("r", encoding="utf-8") as f:
            return [canonicalize(parse_label(line)) for line in f if line.strip()]
    if fmt == "jsonl":
        rows = load_jsonl(path)
        return [canonicalize(parse_label(str(row[field]))) for row in rows]
    if fmt == "json":
        data = load_json(path)
        if not isinstance(data, list):
            raise ValueError("prediction json must be a JSON list")
        if data and isinstance(data[0], dict):
            return [canonicalize(parse_label(str(item[field]))) for item in data]
        return [canonicalize(parse_label(str(item))) for item in data]
    raise ValueError(f"Unsupported prediction format: {fmt}")


def main() -> None:
    args = parse_args()
    ref_path = Path(args.reference)
    pred_path = Path(args.prediction)
    gold = load_reference(ref_path, args.reference_format)
    pred = load_predictions(pred_path, args.prediction_format, args.prediction_field)

    if len(gold) != len(pred):
        raise ValueError(f"Length mismatch: gold={len(gold)} pred={len(pred)}")

    total = len(gold)
    correct = sum(1 for g, p in zip(gold, pred) if g == p)
    acc = correct / total if total else 0.0
    report = {
        "total": total,
        "correct": correct,
        "exact_match": acc,
        "reference": str(ref_path),
        "prediction": str(pred_path),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output_report:
        out = Path(args.output_report)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
