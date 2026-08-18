from pathlib import Path

import pytest


QWEN_TOKENIZER_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"


@pytest.mark.verl
def test_verl_multiturn_dataset_reads_exported_messages(tmp_path: Path):
    pytest.importorskip("verl")
    pytest.importorskip("transformers")
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer
    from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset

    from agent.training.sft import export_sft_dataset

    fixture_dir = Path(__file__).parent / "fixtures"
    output_dir = tmp_path / "sft"
    from agent.task.canonical_dataset import load_corpus_categories

    corpus = {
        category.category_id: category
        for category in load_corpus_categories(fixture_dir / "corpus.json")
    }
    export_sft_dataset(
        fixture_dir / "canonical" / "all.json",
        fixture_dir,
        output_dir,
        fixture_dir / "registry.json",
        fixture_dir / "task.json",
        corpus=corpus,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        revision=QWEN_TOKENIZER_REVISION,
    )
    config = OmegaConf.create(
        {
            "messages_key": "messages",
            "tools_key": "tools",
            "enable_thinking_key": "enable_thinking",
            "enable_thinking_default": None,
            "pad_mode": "no_padding",
            "max_length": 512,
            "truncation": "error",
            "apply_chat_template_kwargs": {},
            "ignore_input_ids_mismatch": False,
        }
    )

    dataset = MultiTurnSFTDataset(
        str(output_dir / "train.parquet"),
        tokenizer,
        config,
        processor=None,
        max_samples=-1,
    )
    item = dataset[0]

    assert len(dataset) == 2
    assert item["input_ids"].numel() > 0
    assert item["loss_mask"].sum().item() > 0
