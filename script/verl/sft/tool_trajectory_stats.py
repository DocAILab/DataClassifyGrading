"""Tool-trajectory SFT rendering snapshot and token-budget inspection.

Follows the existing prompt_stats contract (``script.verl.sft.prompt_stats``):
the chat template is applied to the FULL messages list with
``add_generation_prompt=False``, character stats always run, and token stats
run when a real tokenizer is available (``--model``).

Two outputs:

- ``--snapshot``: a deterministic JSON snapshot of the chat-template RENDERED
  STRING for the first ``--snapshot-rows`` rows of every (split, class)
  combination. Used to pin qwen3.5 chat-template rendering for the tool-loop
  SFT data; byte-stable given the same tokenizer and parquet.
- the budget report: per-split/class char stats, token stats (with a
  tokenizer), and rows exceeding ``--max-length``.

This is a detection and reporting tool only — it never changes prompts,
messages, or the trajectory design.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any
import sys

from agent.training.sft.tool_trajectories import SPLITS, TRAJECTORY_CLASSES

REPORT_FORMAT = "verl_tool_trajectory_stats_v1"


def _token_length(tokenized: Any) -> int:
    if isinstance(tokenized, dict):
        tokenized = tokenized.get("input_ids")
    if tokenized is None:
        raise ValueError("tokenizer did not return input_ids")
    if hasattr(tokenized, "numel"):
        return int(tokenized.numel())
    if hasattr(tokenized, "tolist"):
        tokenized = tokenized.tolist()
    if isinstance(tokenized, list) and tokenized and isinstance(tokenized[0], list):
        if len(tokenized) != 1:
            raise ValueError("tokenizer returned an unexpected batched result")
        tokenized = tokenized[0]
    return len(tokenized)


def _char_lengths(rows: list[dict[str, Any]]) -> list[int]:
    lengths = []
    for row in rows:
        messages = row.get("messages")
        if not isinstance(messages, list):
            continue
        lengths.append(
            sum(len(str(message.get("content", ""))) for message in messages)
        )
    return lengths


def _stats(lengths: list[int]) -> dict[str, Any]:
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


def inspect_tool_trajectory_stats(
    dataset_dir: str | Path,
    tokenizer=None,
    *,
    max_length: int | None = None,
    snapshot_rows: int = 2,
) -> dict[str, Any]:
    """Measure rendered lengths and optionally snapshot rendered strings."""

    if max_length is not None and max_length <= 0:
        raise ValueError("max_length must be positive")
    if snapshot_rows < 0:
        raise ValueError("snapshot_rows must be non-negative")
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - dependency install issue
        raise RuntimeError("tool-trajectory stats require pyarrow") from exc

    root = Path(dataset_dir)
    report: dict[str, Any] = {
        "format": REPORT_FORMAT,
        "rendering": "chat-template full messages, add_generation_prompt=False",
        "unit": "chars",
        "tokenizer_available": tokenizer is not None,
        "max_length": max_length,
        "snapshot_rows_per_class": snapshot_rows,
        "splits": {},
        "snapshot": {},
    }
    for split in SPLITS:
        path = root / f"{split}.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"missing parquet split: {path}")
        rows = pq.read_table(path).to_pylist()
        if not rows:
            raise ValueError(f"parquet split is empty: {path}")

        class_rows: dict[str, list[dict[str, Any]]] = {name: [] for name in TRAJECTORY_CLASSES}
        for row in rows:
            trajectory_class = row.get("trajectory_class")
            if trajectory_class in class_rows:
                class_rows[trajectory_class].append(row)

        details: dict[str, Any] = {"rows": len(rows), "classes": {}}
        over_limit: list[dict[str, Any]] = []
        for trajectory_class, class_rows_for_class in class_rows.items():
            char_stats = _stats(_char_lengths(class_rows_for_class))
            stats: dict[str, Any] = {"chars": char_stats}
            if tokenizer is not None and char_stats["rows"]:
                token_lengths = []
                for row in class_rows_for_class:
                    messages = row.get("messages")
                    if not isinstance(messages, list):
                        continue
                    tokenized = tokenizer.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=False,
                    )
                    token_lengths.append(_token_length(tokenized))
                token_stats = _stats(token_lengths)
                stats["tokens"] = token_stats
                if max_length is not None and token_stats["max"] > max_length:
                    over_limit.append(
                        {
                            "trajectory_class": trajectory_class,
                            "max_tokens": token_stats["max"],
                            "max_length": max_length,
                        }
                    )
            details["classes"][trajectory_class] = stats
        if tokenizer is not None:
            if max_length is not None:
                all_lengths = []
                for trajectory_class in TRAJECTORY_CLASSES:
                    all_lengths.extend(
                        _token_length(
                            tokenizer.apply_chat_template(
                                row.get("messages"),
                                tokenize=True,
                                add_generation_prompt=False,
                            )
                        )
                        for row in class_rows[trajectory_class]
                    )
                split_max = max(all_lengths) if all_lengths else 0
                if split_max > max_length:
                    over_limit.append(
                        {
                            "trajectory_class": "any",
                            "max_tokens": split_max,
                            "max_length": max_length,
                        }
                    )
        details["over_limit"] = over_limit
        report["splits"][split] = details

        if tokenizer is not None and snapshot_rows > 0:
            snapshot: dict[str, list[dict[str, Any]]] = {}
            for trajectory_class in TRAJECTORY_CLASSES:
                entries = []
                for row in class_rows[trajectory_class][:snapshot_rows]:
                    messages = row.get("messages")
                    if not isinstance(messages, list):
                        continue
                    rendered = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=False,
                    )
                    if not isinstance(rendered, str):
                        raise ValueError("tokenize=False must return a rendered string")
                    entries.append(
                        {
                            "source_id": row.get("source_id"),
                            "trajectory_class": trajectory_class,
                            "rendered": rendered,
                        }
                    )
                snapshot[trajectory_class] = entries
            report["snapshot"][split] = snapshot
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument(
        "--model",
        help="Optional HF model id/path for tokenizer-based rendering and token stats",
    )
    parser.add_argument("--max-length", type=int, help="Context limit to check against")
    parser.add_argument(
        "--snapshot-rows",
        type=int,
        default=2,
        help="Rows per split/class to snapshot (requires --model)",
    )
    parser.add_argument("--report", help="Optional JSON output path")
    args = parser.parse_args(argv)

    tokenizer = None
    try:
        if args.model:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(args.model)
        report = inspect_tool_trajectory_stats(
            args.dataset_dir,
            tokenizer,
            max_length=args.max_length,
            snapshot_rows=args.snapshot_rows,
        )
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        print(f"tool_trajectory_stats: {exc}", file=sys.stderr)
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
