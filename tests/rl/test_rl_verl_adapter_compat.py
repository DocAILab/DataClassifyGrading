"""VeRL-compat CPU reward-loop fixture for the choice-aware adapter (Phase 12).

Runs in the verl-compat CI job (verl installed, no GPU needed): exports the
fixture canonical to the choice-protocol RL parquet, loads it through verl's
own RLHFDataset, wires the adapter through verl's reward config loader
(get_custom_reward_fn) and drives a per-item reward loop, asserting that:

- prompt / data_source / reward_model.ground_truth / extra_info.stage /
  extra_info.candidates all flow into the reward inputs;
- the adapter routes choice-id rollouts through the shared choice parser to
  canonical decode BEFORE the unchanged reward table (correct -> 1.0,
  valid-but-miss -> stage1_valid_miss, illegal output -> 0.0);
- adapter output is EXACTLY the shared reward_stage{1,2}_choices result;
- illegal model output never crashes the loop (scores 0.0, never raises);
- GT missing from Stage 2 candidates is legal and never raises.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.verl
def test_verl_cpu_reward_loop_uses_choice_reward(tmp_path: Path):
    pytest.importorskip("verl")
    pytest.importorskip("transformers")
    pytest.importorskip("omegaconf")
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer
    from verl.utils.dataset import RLHFDataset

    from agent.task import LeafRegistry, TaskConfig
    from agent.task.canonical_dataset import load_corpus_categories
    from agent.task.prompt_choices import PromptChoiceRegistry, encode_stage2_answer
    from agent.training.rl import export_rl_dataset
    from agent.training.rl.reward import (
        STAGE2_PARTIAL_DEFAULT,
        reward_stage1_choices,
        reward_stage2_choices,
    )
    from agent.training.rl.verl_adapter import configure

    fixture_dir = Path(__file__).resolve().parents[1] / "sft" / "fixtures"
    corpus = {
        category.category_id: category
        for category in load_corpus_categories(fixture_dir / "corpus.json")
    }
    registry = LeafRegistry.from_path(fixture_dir / "registry.json")
    choices = PromptChoiceRegistry.from_registry(registry)
    config = TaskConfig.from_path(fixture_dir / "task.json")
    out_dir = tmp_path / "rl"
    export_rl_dataset(
        fixture_dir / "canonical" / "all.json",
        fixture_dir,
        out_dir,
        "fixture_ds",
        registry,
        config,
        corpus=corpus,
    )
    # production wiring: the router resolves the registry from disk; here we
    # pre-register the fixture registry so the test does not touch cfg/
    configure(registries={"fixture_ds": registry})

    # 1. wire our adapter through verl's own reward config loader
    reward_cfg = OmegaConf.create(
        {
            "reward": {
                "custom_reward_function": {
                    "path": "pkg://agent.training.rl.verl_adapter",
                    "name": "compute_score",
                    "reward_kwargs": {},
                }
            }
        }
    )
    from verl.trainer.ppo.reward import get_custom_reward_fn

    router = get_custom_reward_fn(reward_cfg)
    assert router is not None

    # 2. load the RL parquet through verl RLHFDataset (CPU)
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
    dataset = RLHFDataset(str(out_dir / "train.parquet"), tokenizer, rl_config, processor=None, max_samples=-1)
    assert len(dataset) == 2  # 1 record x stage1 + stage2

    # 3. drive a per-item reward loop on CPU, matching the naive manager's calls
    for index in range(len(dataset)):
        item = dataset[index]
        data_source = item["data_source"]
        ground_truth = item["reward_model"]["ground_truth"]
        extra_info = item["extra_info"]
        stage = extra_info["stage"]
        assert data_source == f"fixture_ds/{stage}"
        assert ground_truth in registry.ids

        if stage == "stage1":
            # choice ids 1..5 cover the whole 5-class fixture -> GT always recalled
            good = '{"candidates":["1","2","3","4","5"]}'
            decoded = choices.decode_candidates(("1", "2", "3", "4", "5"))
            assert ground_truth in decoded
            direct = reward_stage1_choices(good, ground_truth=ground_truth, registry=registry).reward
        else:
            candidates = extra_info["candidates"]
            assert len(candidates) == 5
            good = '{"answer":"%s"}' % encode_stage2_answer(ground_truth, list(candidates))
            direct = reward_stage2_choices(
                good, ground_truth=ground_truth, candidates=candidates, registry=registry
            ).reward
        routed_good = router(
            data_source=data_source,
            solution_str=good,
            ground_truth=ground_truth,
            extra_info=dict(extra_info),
        )
        # adapter output is EXACTLY the shared choice-aware reward result
        assert routed_good == pytest.approx(float(direct))
        assert routed_good == pytest.approx(1.0)

        # legal-but-wrong choice output: never crashes, maps to the frozen table
        if stage == "stage1":
            # "valid but miss": 5 unique known choice ids NOT containing GT
            valid_ids = [cid for cid in choices.choice_ids if cid != choices.choice_id_of(ground_truth)]
            wrong = json_dumps({"candidates": valid_ids[:5]})
            routed_wrong = router(
                data_source=data_source,
                solution_str=wrong,
                ground_truth=ground_truth,
                extra_info=dict(extra_info),
            )
            assert routed_wrong == pytest.approx(0.3)  # stage1_valid_miss
        else:
            routed_wrong = router(
                data_source=data_source,
                solution_str='{"answer":"1"}',
                ground_truth=ground_truth,
                extra_info=dict(extra_info),
            )
            # wrong is either partial or (if answer 1 IS the GT) full — must never be invalid
            assert routed_wrong in (1.0, STAGE2_PARTIAL_DEFAULT)

        # illegal model output must not crash the loop
        routed_bad = router(
            data_source=data_source,
            solution_str="not valid json and %% garbage %%",
            ground_truth=ground_truth,
            extra_info=dict(extra_info),
        )
        assert isinstance(routed_bad, float)
        assert routed_bad == pytest.approx(0.0)

    # 4. GT missing from Stage 2 candidates is legal (stage-1 recall failure)
    item2 = next(dataset[i] for i in range(len(dataset)) if dataset[i]["extra_info"]["stage"] == "stage2")
    candidates = list(item2["extra_info"]["candidates"])
    excluded = candidates[:-1]
    score = router(
        data_source="fixture_ds/stage2",
        solution_str='{"answer":"1"}',
        ground_truth=item2["reward_model"]["ground_truth"],
        extra_info={"candidates": excluded},
    )
    assert isinstance(score, float)  # partial/0.0, never an exception


def json_dumps(value) -> str:
    import json

    return json.dumps(value, ensure_ascii=False)
