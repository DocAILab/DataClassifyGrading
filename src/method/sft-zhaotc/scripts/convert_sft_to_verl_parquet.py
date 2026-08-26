#!/usr/bin/env python3
"""Convert one-turn SFT JSON files to VeRL MultiTurnSFTDataset parquet files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert SFT JSON to VeRL parquet.")
    parser.add_argument("--input", required=True, help="ShareGPT or Alpaca JSON file.")
    parser.add_argument("--output", required=True, help="Output parquet path.")
    parser.add_argument(
        "--format",
        choices=["auto", "sharegpt", "alpaca"],
        default="auto",
        help="Input SFT format.",
    )
    parser.add_argument(
        "--enable-thinking",
        choices=["true", "false", "omit"],
        default="false",
        help="Value for VeRL enable_thinking column. Use false for Qwen3 classification SFT.",
    )
    return parser.parse_args()


def load_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list: {path}")
    return data


def detect_format(records: list[dict[str, Any]]) -> str:
    if not records:
        raise ValueError("Empty dataset")
    first = records[0]
    if "conversations" in first:
        return "sharegpt"
    if {"instruction", "input", "output"} <= set(first):
        return "alpaca"
    raise ValueError("Cannot detect SFT format")


def sharegpt_to_messages(item: dict[str, Any]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    system = str(item.get("system", "")).strip()
    if system:
        messages.append({"role": "system", "content": system})
    for message in item.get("conversations", []):
        role = message.get("from")
        if role == "human":
            role = "user"
        elif role == "gpt":
            role = "assistant"
        messages.append({"role": str(role), "content": str(message.get("value", ""))})
    return messages


def alpaca_to_messages(item: dict[str, Any]) -> list[dict[str, str]]:
    instruction = str(item.get("instruction", "")).strip()
    user_input = str(item.get("input", "")).strip()
    output = str(item.get("output", "")).strip()
    return [
        {"role": "system", "content": instruction},
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": output},
    ]


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    records = load_json(input_path)
    fmt = detect_format(records) if args.format == "auto" else args.format

    rows: list[dict[str, Any]] = []
    for item in records:
        if fmt == "sharegpt":
            messages = sharegpt_to_messages(item)
        else:
            messages = alpaca_to_messages(item)
        row: dict[str, Any] = {
            "id": str(item.get("id", "")),
            "messages": messages,
        }
        if args.enable_thinking != "omit":
            row["enable_thinking"] = args.enable_thinking == "true"
        rows.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output_path, index=False)
    print(f"input: {input_path}")
    print(f"format: {fmt}")
    print(f"records: {len(rows)}")
    print(f"wrote: {output_path}")


if __name__ == "__main__":
    main()
