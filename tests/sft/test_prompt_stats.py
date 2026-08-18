"""Tests for script.verl.sft.prompt_stats (character + tokenizer stats)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.training.sft import export_sft_dataset
from script.verl.sft.prompt_stats import inspect_prompt_lengths


class FakeTokenizer:
    """Mimics transformers BatchEncoding return shape."""

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        content = "".join(str(message.get("content", "")) for message in messages)
        if not tokenize:
            return content
        return {"input_ids": list(range(len(content)))}


def _export_fixture(tmp_path: Path) -> Path:
    fixture_dir = Path(__file__).parent / "fixtures"
    out = tmp_path / "sft"
    from agent.task.canonical_dataset import load_corpus_categories

    corpus = {
        category.category_id: category
        for category in load_corpus_categories(fixture_dir / "corpus.json")
    }
    export_sft_dataset(
        fixture_dir / "canonical" / "all.json",
        fixture_dir,
        out,
        fixture_dir / "registry.json",
        fixture_dir / "task.json",
        corpus=corpus,
    )
    return out


def test_character_stats_per_split_and_stage(tmp_path: Path) -> None:
    out = _export_fixture(tmp_path)
    report = inspect_prompt_lengths(out)
    assert report["tokenizer_available"] is False
    train_stage1 = report["splits"]["train"]["stage1"]
    assert train_stage1["rows"] > 0
    assert 0 < train_stage1["p95"] <= train_stage1["max"]
    assert train_stage1["mean"] > 0
    assert report["splits"]["train"]["stage2"]["max"] > 0


def test_tokenizer_stats_with_batch_encoding_shape(tmp_path: Path) -> None:
    out = _export_fixture(tmp_path)
    report = inspect_prompt_lengths(out, FakeTokenizer(), max_length=1_000_000)
    assert report["tokenizer_available"] is True
    tokens = report["splits"]["train"]["stage1"]["tokens"]
    chars = report["splits"]["train"]["stage1"]["chars"]
    assert tokens["rows"] == chars["rows"]
    assert tokens["max"] == chars["max"]  # fake tokenizer: one token per char
    assert report["exceeds_max_length"] == []


def test_exceeds_max_length_is_reported(tmp_path: Path) -> None:
    out = _export_fixture(tmp_path)
    report = inspect_prompt_lengths(out, FakeTokenizer(), max_length=10)
    assert report["exceeds_max_length"]
    assert {item["stage"] for item in report["exceeds_max_length"]} == {"stage1", "stage2"}
    assert {item["split"] for item in report["exceeds_max_length"]} == {"train", "val", "test"}


def test_missing_split_fails_fast(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="missing parquet split"):
        inspect_prompt_lengths(tmp_path)
