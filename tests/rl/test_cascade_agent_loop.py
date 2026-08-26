"""CPU-visible contract for the VeRL 0.8 direct-cascade AgentLoop adapter."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest

from agent.task import GradingConfig, LeafRegistry, TaskConfig
from agent.task.assets import load_corpus_categories
from agent.task.prompts import build_stage2_prompt
from script.verl.rl.cascade_agent_loop import (
    VERL_AGENT_LOOP_AVAILABLE,
    DataClassifyCascadeAgentLoop,
    build_stage2_bridge_text,
)
from agent.task.grading_manifest import DatasetGradingManifest


FIXTURES = Path(__file__).resolve().parents[1] / "sft" / "fixtures"


def test_stage2_bridge_contains_shared_strict_prompt() -> None:
    registry = LeafRegistry.from_path(FIXTURES / "registry.json")
    corpus = {
        item.category_id: item
        for item in load_corpus_categories(FIXTURES / "corpus.json")
    }
    candidates = tuple(registry.ids[:5])
    prompt = build_stage2_prompt(
        {"field_name": "abc_id"},
        candidates,
        registry,
        TaskConfig(("field_name",)),
        corpus=corpus,
        grading=GradingConfig(("L1", "L2", "L3", "L4")),
    )
    bridge = build_stage2_bridge_text(prompt)
    assert prompt.system in bridge
    assert prompt.user in bridge
    assert 'keys "answer" and "level"' in bridge
    assert "database_name" not in bridge
    assert "table_name" not in bridge


def test_missing_verl_fails_at_adapter_construction_not_module_import() -> None:
    if VERL_AGENT_LOOP_AVAILABLE:
        pytest.skip("VeRL installed; subclass compatibility is covered by verl-compat CI")
    with pytest.raises(RuntimeError, match="VeRL 0.8"):
        DataClassifyCascadeAgentLoop()


@pytest.mark.verl
def test_real_adapter_reuses_sticky_request_and_masks_bridge() -> None:
    """Probe the official 0.8 output seam without starting Ray/vLLM/GPU."""

    if not VERL_AGENT_LOOP_AVAILABLE:
        pytest.skip("VeRL 0.8 is not installed")
    registry = LeafRegistry.from_path(FIXTURES / "registry.json")
    corpus = {
        item.category_id: item
        for item in load_corpus_categories(FIXTURES / "corpus.json")
    }
    manifest = DatasetGradingManifest.from_path(FIXTURES / "grading_manifest.json")

    class TokenOutput:
        def __init__(self, token_ids, log_probs):
            self.token_ids = token_ids
            self.log_probs = log_probs
            self.num_preempted = 0

    class Server:
        def __init__(self):
            self.calls = []
            self.outputs = [
                TokenOutput([1], [-0.1]),
                TokenOutput([2], [-0.2]),
            ]

        async def generate(self, *, request_id, prompt_ids, sampling_params, **_):
            self.calls.append((request_id, list(prompt_ids), dict(sampling_params)))
            return self.outputs.pop(0)

    class Tokenizer:
        def decode(self, token_ids, *, skip_special_tokens=True):
            del skip_special_tokens
            return (
                '{"candidates":["1","2","3","4","5"]}'
                if list(token_ids) == [1]
                else '{"answer":"1","level":"L2"}'
            )

    loop = object.__new__(DataClassifyCascadeAgentLoop)
    loop.registry = registry
    loop.corpus = corpus
    loop.grading_manifest = manifest
    loop.stage1_max_tokens = 3
    loop.stage2_max_tokens = 2
    loop.response_length = 16
    loop.tokenizer = Tokenizer()
    loop.server_manager = Server()

    async def apply_chat_template(self, messages, **kwargs):
        del self, messages
        return [10] if not kwargs.get("remove_system_prompt") else [20, 21]

    loop.apply_chat_template = MethodType(apply_chat_template, loop)
    output = asyncio.run(
        loop.run(
            {"temperature": 0.0},
            raw_prompt=[{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
            extra_info={
                "dataset": "finance",
                "stage": "stage1",
                "source_id": "source-1",
                "ground_truth_level": "L2",
                "metadata": {"field_name": "f", "table_name": "t"},
            },
            reward_model={"style": "rule", "ground_truth": "demo:alpha"},
        )
    )

    assert len(loop.server_manager.calls) == 2
    assert loop.server_manager.calls[0][0] == loop.server_manager.calls[1][0]
    assert [call[2]["max_tokens"] for call in loop.server_manager.calls] == [3, 2]
    assert output.response_ids == [1, 20, 21, 2]
    assert output.response_mask == [1, 0, 0, 1]
    assert output.response_logprobs == [-0.1, 0.0, 0.0, -0.2]
    assert output.reward_score == pytest.approx(1.0)


@pytest.mark.verl
def test_real_adapter_rejects_generation_over_budget() -> None:
    if not VERL_AGENT_LOOP_AVAILABLE:
        pytest.skip("VeRL 0.8 is not installed")
    loop = object.__new__(DataClassifyCascadeAgentLoop)
    with pytest.raises(ValueError, match="budget"):
        loop._checked_tokens(SimpleNamespace(token_ids=[1, 2, 3]), stage="Stage 1", budget=2)
