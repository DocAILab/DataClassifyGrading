"""Stage-1/Stage-2 prompt length statistics for exported SFT parquet.

Character-based stats always run (max / p95 / mean per split and stage).
When a real tokenizer is available (--model, e.g. Qwen/Qwen2.5-7B-Instruct)
the chat-template token count is measured as well. This is a detection and
reporting tool only — it never changes prompts or the retrieval design.

Usage:
    python -m script.verl.sft.prompt_stats --dataset-dir <parquet dir> [--model ...] [--report out.json]
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Mapping
import sys

SPLITS = ("train", "val", "test")
STAGES = ("stage1", "stage2")


def _lengths(rows: list[dict], stage: str) -> list[int]:
    lengths = []
    for row in rows:
        if row.get("stage") != stage:
            continue
        messages = row.get("messages")
        if not isinstance(messages, list):
            continue
        lengths.append(
            sum(len(str(message.get("content", ""))) for message in messages)
        )
    return lengths


def _stats(lengths: list[int]) -> dict:
    if not lengths:
        return {"rows": 0}
    ordered = sorted(lengths)
    p95_index = max(0, int(0.95 * len(ordered)) - 1)
    return {
        "rows": len(lengths),
        "max": ordered[-1],
        "p95": ordered[p95_index],
        "mean": round(statistics.mean(ordered), 1),
    }


def inspect_prompt_lengths(
    dataset_dir: str | Path,
    tokenizer=None,
    *,
    max_length: int | None = None,
) -> dict[str, Any]:
    import pyarrow.parquet as pq

    root = Path(dataset_dir)
    report: dict[str, Any] = {
        "unit": "chars",
        "tokenizer_available": tokenizer is not None,
        "splits": {},
        "exceeds_max_length": [],
    }
    for split in SPLITS:
        path = root / f"{split}.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"missing parquet split: {path}")
        rows = pq.read_table(path).to_pylist()
        details: dict[str, Any] = {}
        for stage in STAGES:
            char_lengths = _lengths(rows, stage)
            stats = _stats(char_lengths)
            if tokenizer is not None and stats["rows"]:
                token_lengths = [
                    len(
                        _token_ids(
                            tokenizer,
                            row["messages"],
                        )
                    )
                    for row in rows
                    if row.get("stage") == stage
                ]
                token_stats = _stats(token_lengths)
                stats = {
                    "chars": stats,
                    "tokens": token_stats,
                }
                if max_length is not None and token_stats["max"] > max_length:
                    report["exceeds_max_length"].append(
                        {
                            "split": split,
                            "stage": stage,
                            "max_tokens": token_stats["max"],
                            "max_length": max_length,
                        }
                    )
            details[stage] = stats
        report["splits"][split] = details
    return report


def _token_ids(tokenizer, messages) -> list[int]:
    tokenized = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
    )
    # transformers BatchEncoding is a UserDict, not a dict subclass
    if isinstance(tokenized, Mapping):
        tokenized = tokenized.get("input_ids")
    if tokenized is None:
        raise ValueError("tokenizer did not return input_ids")
    if hasattr(tokenized, "tolist"):
        tokenized = tokenized.tolist()
    if isinstance(tokenized, list) and tokenized and isinstance(tokenized[0], list):
        if len(tokenized) != 1:
            raise ValueError("unexpected batched tokenization")
        tokenized = tokenized[0]
    return list(tokenized)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument(
        "--model",
        help="Optional HF model id/path for tokenizer-based token statistics",
    )
    parser.add_argument("--max-length", type=int, help="Context limit to check against")
    parser.add_argument("--report", help="Optional JSON output path")
    args = parser.parse_args(argv)

    tokenizer = None
    try:
        if args.model:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(args.model)
        report = inspect_prompt_lengths(
            args.dataset_dir,
            tokenizer,
            max_length=args.max_length,
        )
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        print(f"prompt_stats: {exc}", file=sys.stderr)
        return 2
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report:
        path = Path(args.report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered, encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
