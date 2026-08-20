"""Model-specific token-budget inspection for exported SFT messages."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


PRODUCTION_SPLITS = ("train", "val")


def _token_length(tokenized: Any) -> int:
    if isinstance(tokenized, Mapping):
        tokenized = tokenized.get("input_ids")
    if tokenized is None:
        raise ValueError("tokenizer did not return input_ids")
    if hasattr(tokenized, "numel"):
        return int(tokenized.numel())
    if (
        isinstance(tokenized, list)
        and tokenized
        and isinstance(tokenized[0], list)
    ):
        if len(tokenized) != 1:
            raise ValueError("tokenizer returned an unexpected batched result")
        tokenized = tokenized[0]
    return len(tokenized)


def inspect_token_budget(
    dataset_dir: str | Path,
    tokenizer: Any,
    *,
    max_length: int,
    splits: tuple[str, ...] | list[str] = PRODUCTION_SPLITS,
) -> dict[str, Any]:
    """Measure chat-template lengths and report rows exceeding ``max_length``."""
    if max_length <= 0:
        raise ValueError("max_length must be positive")
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - dependency installation issue
        raise RuntimeError("token-budget inspection requires pyarrow") from exc

    requested = tuple(splits)
    if not requested or any(split not in PRODUCTION_SPLITS for split in requested):
        raise ValueError("splits must be a non-empty subset of: train, val")
    if len(set(requested)) != len(requested):
        raise ValueError("split names must be unique")
    root = Path(dataset_dir)
    report: dict[str, Any] = {
        "valid": True,
        "max_length": max_length,
        "requested_splits": list(requested),
        "real_test_split_read": False,
        "splits": {},
    }
    for split in requested:
        path = root / f"{split}.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"missing parquet split: {path}")
        rows = pq.read_table(path).to_pylist()
        if not rows:
            raise ValueError(f"parquet split is empty: {path}")

        longest_tokens = -1
        longest_source_id = None
        over_limit = 0
        for row in rows:
            messages = row.get("messages")
            if not isinstance(messages, list):
                raise ValueError(f"row in {path} has invalid messages")
            tokenized = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
            )
            length = _token_length(tokenized)
            if length > longest_tokens:
                longest_tokens = length
                longest_source_id = row.get("source_id")
            if length > max_length:
                over_limit += 1

        report["splits"][split] = {
            "rows": len(rows),
            "max_tokens": longest_tokens,
            "longest_source_id": longest_source_id,
            "over_limit": over_limit,
        }
        if over_limit:
            report["valid"] = False
    return report
