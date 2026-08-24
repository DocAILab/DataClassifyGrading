"""VeRL 0.8 compatibility seam for the exported SFT parquet contract."""

from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path

import pytest

from agent.task import GradingConfig, LeafRegistry, TaskConfig
from agent.task.assets import load_corpus_categories
from agent.training.sft import export_sft_dataset


pytestmark = pytest.mark.verl
ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "sft" / "fixtures"


def _tiny_tokenizer():
    """Build a local tokenizer so compat tests never download a model."""

    pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    from transformers import PreTrainedTokenizerFast

    vocab = {
        "<unk>": 0,
        "<pad>": 1,
        "<bos>": 2,
        "<eos>": 3,
        "system": 4,
        "user": 5,
        "assistant": 6,
    }
    tokenizer = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
    )
    fast.chat_template = (
        "{% for message in messages %}"
        "{{ message['role'] }}: {{ message['content'] }}\\n"
        "{% endfor %}"
        "{% if add_generation_prompt %}assistant: {% endif %}"
    )
    return fast


def _export(tmp_path: Path) -> Path:
    registry = LeafRegistry.from_path(FIXTURES / "registry.json")
    task = TaskConfig.from_path(FIXTURES / "task.json")
    corpus = {
        category.category_id: category
        for category in load_corpus_categories(FIXTURES / "corpus.json")
    }
    export_sft_dataset(
        FIXTURES / "canonical" / "all.json",
        None,
        tmp_path,
        registry,
        task,
        corpus=corpus,
        grading=GradingConfig.from_path(FIXTURES / "grading.json"),
    )
    return tmp_path


def test_exported_parquet_loads_through_verl_08_multiturn_dataset(tmp_path) -> None:
    # importorskip is intentional: the ordinary CPU unit suite has no VeRL;
    # the verl-compat CI job installs the pinned wheel and executes this test.
    verl = pytest.importorskip("verl")
    assert importlib.metadata.version("verl") == "0.8.0"
    dataset_cls = pytest.importorskip(
        "verl.utils.dataset.multiturn_sft_dataset"
    ).MultiTurnSFTDataset

    output = _export(tmp_path / "release")
    tokenizer = _tiny_tokenizer()
    config = {
        "messages_key": "messages",
        "pad_mode": "no_padding",
        "truncation": "error",
        "max_length": 512,
        "num_workers": 0,
    }
    for split in ("train", "val", "test"):
        dataset = dataset_cls(str(output / f"{split}.parquet"), tokenizer, config)
        assert len(dataset) == 6
        assert all(
            [message["role"] for message in messages]
            == ["system", "user", "assistant"]
            for messages in dataset.messages
        )

    # Keep the fixture contract visible to reviewers and future compatibility
    # updates: the joint stage2 assistant contains both answer and level.
    import pyarrow.parquet as pq

    rows = pq.read_table(output / "train.parquet").to_pylist()
    stage2 = next(row for row in rows if row["stage"] == "stage2")
    assert set(json.loads(stage2["messages"][-1]["content"])) == {"answer", "level"}
