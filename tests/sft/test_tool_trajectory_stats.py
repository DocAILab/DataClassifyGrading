"""Rendering-snapshot and token-budget tests with a deterministic stub."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from script.verl.sft.tool_trajectory_stats import inspect_tool_trajectory_stats


ROOT = Path(__file__).resolve().parents[2]


class StubTokenizer:
    """Deterministic fake tokenizer: im_start-style rendering, stable counts."""

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ):
        rendered = "".join(
            f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
            for message in messages
        )
        if add_generation_prompt:
            rendered += "<|im_start|>assistant\n"
        if not tokenize:
            return rendered
        length = len(rendered) // 4 + 1
        return {"input_ids": [length] * length}


@pytest.fixture
def dataset_dir(tmp_path: Path) -> Path:
    from agent.task import GradingConfig, LeafRegistry, TaskConfig
    from agent.task.assets import load_corpus_categories
    from agent.training.sft.tool_trajectories import export_tool_trajectory_dataset

    output = tmp_path / "release"
    export_tool_trajectory_dataset(
        ROOT / "tests" / "sft" / "fixtures" / "tool_trajectory_canonical.json",
        output,
        LeafRegistry.from_path(ROOT / "tests" / "sft" / "fixtures" / "tool_registry.json"),
        corpus={
            category.category_id: category
            for category in load_corpus_categories(
                ROOT / "tests" / "sft" / "fixtures" / "tool_corpus.json"
            )
        },
        task_config=TaskConfig.from_mapping(
            {
                "task_name": "synthetic_field_classification",
                "metadata_fields": [
                    "field_name",
                    "table_name",
                    "field_description",
                    "table_description",
                ],
            }
        ),
        grading=GradingConfig.from_path(
            ROOT / "tests" / "sft" / "fixtures" / "grading.json"
        ),
    )
    return output


def test_snapshot_covers_every_split_and_class_deterministically(
    dataset_dir: Path,
) -> None:
    tokenizer = StubTokenizer()
    first = inspect_tool_trajectory_stats(dataset_dir, tokenizer, snapshot_rows=2)
    second = inspect_tool_trajectory_stats(dataset_dir, tokenizer, snapshot_rows=2)
    assert first == second
    for split in ("train", "val", "test"):
        snapshot = first["snapshot"][split]
        assert set(snapshot) == {"direct", "single_tool", "multi_tool", "no_result"}
        for entries in snapshot.values():
            assert 1 <= len(entries) <= 2
            for entry in entries:
                assert entry["rendered"].startswith("<|im_start|>system")
                assert "<tool_call>" in entry["rendered"] or (
                    entry["trajectory_class"] == "direct"
                    and "<tool_call>" not in entry["rendered"]
                )


def test_budget_reports_char_and_token_stats(dataset_dir: Path) -> None:
    tokenizer = StubTokenizer()
    report = inspect_tool_trajectory_stats(dataset_dir, tokenizer, max_length=10_000)
    assert report["tokenizer_available"] is True
    assert report["splits"]["train"]["classes"]["multi_tool"]["chars"]["max"] > 0
    assert (
        report["splits"]["train"]["classes"]["multi_tool"]["tokens"]["max"] > 0
    )
    assert report["splits"]["train"]["over_limit"] == []


def test_over_limit_detection(dataset_dir: Path) -> None:
    report = inspect_tool_trajectory_stats(dataset_dir, StubTokenizer(), max_length=4)
    assert report["splits"]["train"]["over_limit"]
    assert report["splits"]["val"]["over_limit"]


def test_chars_only_without_tokenizer(dataset_dir: Path) -> None:
    report = inspect_tool_trajectory_stats(dataset_dir, None)
    assert report["tokenizer_available"] is False
    assert report["snapshot"] == {}
    assert "tokens" not in report["splits"]["train"]["classes"]["direct"]
    assert report["splits"]["train"]["classes"]["direct"]["chars"]["rows"] > 0


def test_rejects_nonpositive_max_length(dataset_dir: Path) -> None:
    with pytest.raises(ValueError, match="max_length"):
        inspect_tool_trajectory_stats(dataset_dir, StubTokenizer(), max_length=0)
