"""Verify the RL parquet is loadable by the pinned VeRL v0.8.0 RLHFDataset.

Run in the verl-compat CI job (and locally where verl is installed):
exports the fixture canonical to the five-field RL parquet and loads it
through verl's own RLHFDataset, asserting the reward metadata survives into
the per-item batch and the raw prompt has no assistant gold response.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.verl
def test_verl_rlhfdataset_reads_exported_rl_parquet(tmp_path: Path):
    pytest.importorskip("verl")
    pytest.importorskip("transformers")
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer
    from verl.utils.dataset import RLHFDataset

    from agent.task import LeafRegistry, TaskConfig
    from agent.task.canonical_dataset import load_corpus_categories
    from agent.training.rl import export_rl_dataset

    fixture_dir = Path(__file__).resolve().parents[1] / "sft" / "fixtures"
    corpus = {
        category.category_id: category
        for category in load_corpus_categories(fixture_dir / "corpus.json")
    }
    config = TaskConfig.from_path(fixture_dir / "task.json")
    output_dir = tmp_path / "rl"
    export_rl_dataset(
        fixture_dir / "canonical" / "all.json",
        fixture_dir,
        output_dir,
        "fixture_ds",
        LeafRegistry.from_path(fixture_dir / "registry.json"),
        config,
        corpus=corpus,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        revision="7ae557604adf67be50417f59c2c2f167def9a775",
    )
    rl_config = OmegaConf.create(
        {
            "cache_dir": str(tmp_path / "verl-cache"),
            "prompt_key": "prompt",
            "max_prompt_length": 2048,
            "truncation": "error",
            "apply_chat_template_kwargs": {},
            "mm_processor_kwargs": {},
        }
    )

    dataset = RLHFDataset(
        str(output_dir / "train.parquet"),
        tokenizer,
        rl_config,
        processor=None,
        max_samples=-1,
    )
    assert len(dataset) == 2  # fixture train split: 1 record x stage1 + stage2
    item = dataset[0]

    raw_prompt = item["raw_prompt"]
    assert [message["role"] for message in raw_prompt] in (["system", "user"], ["user"])
    assert not any(message["role"] == "assistant" for message in raw_prompt)
    assert item["data_source"] in {"fixture_ds/stage1", "fixture_ds/stage2"}
    assert item["ability"] == config.task_name
    ground_truth = item["reward_model"]["ground_truth"]
    assert isinstance(ground_truth, str) and ground_truth
    assert item["extra_info"]["dataset"] == "fixture_ds"
    assert item["extra_info"]["stage"] in {"stage1", "stage2"}
    # Stage 2 rows carry the candidate bundle through verl
    for index in range(len(dataset)):
        current = dataset[index]
        if current["extra_info"]["stage"] == "stage2":
            assert len(current["extra_info"]["candidates"]) == 5
