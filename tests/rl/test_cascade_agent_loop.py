"""CPU-visible contract for the VeRL 0.9 native-tool AgentLoop adapter."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.task import GradingConfig, LeafRegistry
from agent.training.rl.native_tools import exact_tool_reward, parse_final_tool_answer
from script.verl.rl.cascade_agent_loop import (
    VERL_AGENT_LOOP_AVAILABLE,
    DataClassifyCascadeAgentLoop,
    _source_from_kwargs,
    final_policy_token_ids,
)


ROOT = Path(__file__).resolve().parents[2]
TRAJECTORY_FIXTURE = Path(__file__).with_name("fixtures") / "native_tool_trajectories.json"
REGISTRY = ROOT / "cfg" / "task" / "leaf_registry.example.json"


def _kwargs(**extra_overrides):
    extra = {
        "dataset": "shougang",
        "stage": "stage1",
        "source_id": "source-1",
        "ground_truth_level": "L2",
        "trajectory_format": "qwen3.5-native-tools-v2",
        "metadata": {
            "field_name": "f",
            "table_name": "t",
            "field_description": "",
            "table_description": "",
        },
    }
    extra.update(extra_overrides)
    return {
        "extra_info": extra,
        "reward_model": {"style": "rule", "ground_truth": "demo:alpha"},
    }


def test_source_contract_is_shougang_native_tools_and_four_metadata_fields() -> None:
    source = _source_from_kwargs(_kwargs())
    assert source.dataset == "shougang"
    assert source.metadata == {
        "field_name": "f",
        "table_name": "t",
        "field_description": "",
        "table_description": "",
    }

    with pytest.raises(ValueError, match="trajectory_format"):
        _source_from_kwargs(_kwargs(trajectory_format="manual-cascade"))
    with pytest.raises(ValueError, match="only shougang"):
        _source_from_kwargs(_kwargs(dataset="finance"))
    with pytest.raises(ValueError, match=r"field_name\+table_name\+field_description\+table_description"):
        _source_from_kwargs(_kwargs(metadata={"field_name": "f"}))


def test_final_policy_segment_supports_no_single_and_multi_tool_paths() -> None:
    assert final_policy_token_ids([1, 2], [1, 1]) == (1, 2)  # no-tool final
    assert final_policy_token_ids([1, 20, 21, 2], [1, 0, 0, 1]) == (2,)
    assert final_policy_token_ids(
        [1, 20, 2, 30, 31, 3, 4],
        [1, 0, 1, 0, 0, 1, 1],
    ) == (3, 4)
    # A trajectory that hits the turn/length cap immediately after a tool
    # observation has no terminal assistant answer and must score zero.
    assert final_policy_token_ids([1, 20], [1, 0]) == ()

    with pytest.raises(ValueError, match="equal length"):
        final_policy_token_ids([1], [1, 0])
    with pytest.raises(ValueError, match="0/1"):
        final_policy_token_ids([1], [2])


def _fixture_response_ids(case: dict) -> tuple[list[int], list[int]]:
    if "response_text" in case:
        return [ord(char) for char in case["response_text"]], [1] * len(case["response_text"])
    ids: list[int] = []
    mask: list[int] = []
    for segment in case["response_segments"]:
        text = segment["text"]
        ids.extend(ord(char) for char in text)
        mask.extend([1 if segment["policy"] else 0] * len(text))
    return ids, mask


@pytest.mark.parametrize(
    "case", json.loads(TRAJECTORY_FIXTURE.read_text(encoding="utf-8")),
    ids=lambda case: case["name"],
)
def test_final_text_rule_and_reward_cover_native_terminal_shapes(case: dict) -> None:
    """Only the trailing policy segment is terminal; malformed text scores zero."""

    response_ids, response_mask = _fixture_response_ids(case)
    final_ids = final_policy_token_ids(response_ids, response_mask)
    final_text = "".join(chr(token) for token in final_ids).strip()
    assert final_text == case["expected_final_text"]

    registry = LeafRegistry.from_path(REGISTRY)
    grading = GradingConfig(("L1", "L2", "L3", "L4"))
    if case["expected_terminal_valid"]:
        parsed = parse_final_tool_answer(final_text, registry=registry, grading=grading)
        assert parsed.category_id == case["ground_truth"]
        assert parsed.level == case["ground_truth_level"]
    else:
        with pytest.raises(ValueError):
            parse_final_tool_answer(final_text, registry=registry, grading=grading)
    assert exact_tool_reward(
        final_text,
        ground_truth=case["ground_truth"],
        ground_truth_level=case["ground_truth_level"],
        registry=registry,
        grading=grading,
    ) == case["expected_reward"]


def test_missing_verl_fails_at_adapter_construction_not_module_import() -> None:
    if VERL_AGENT_LOOP_AVAILABLE:
        pytest.skip("VeRL installed; subclass compatibility is covered separately")
    with pytest.raises(RuntimeError, match="VeRL 0.9.0 native ToolAgentLoop"):
        DataClassifyCascadeAgentLoop()


@pytest.mark.verl
def test_real_adapter_subclasses_official_tool_agent_loop() -> None:
    if not VERL_AGENT_LOOP_AVAILABLE:
        pytest.skip("VeRL 0.9.0 is not installed")
    from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop

    assert issubclass(DataClassifyCascadeAgentLoop, ToolAgentLoop)
    assert hasattr(ToolAgentLoop, "_handle_processing_tools_state")
    assert SimpleNamespace(token_ids=[1]).token_ids == [1]
