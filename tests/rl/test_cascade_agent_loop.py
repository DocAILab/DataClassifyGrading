"""CPU-visible contract for the VeRL 0.9 native-tool AgentLoop adapter."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from script.verl.rl.cascade_agent_loop import (
    VERL_AGENT_LOOP_AVAILABLE,
    DataClassifyCascadeAgentLoop,
    _source_from_kwargs,
    final_policy_token_ids,
)


def _kwargs(**extra_overrides):
    extra = {
        "dataset": "shougang",
        "stage": "stage1",
        "source_id": "source-1",
        "ground_truth_level": "L2",
        "trajectory_format": "qwen3.5-native-tools-v1",
        "metadata": {"field_name": "f", "table_name": "t"},
    }
    extra.update(extra_overrides)
    return {
        "extra_info": extra,
        "reward_model": {"style": "rule", "ground_truth": "demo:alpha"},
    }


def test_source_contract_is_shougang_native_tools_and_field_table_only() -> None:
    source = _source_from_kwargs(_kwargs())
    assert source.dataset == "shougang"
    assert source.metadata == {"field_name": "f", "table_name": "t"}

    with pytest.raises(ValueError, match="trajectory_format"):
        _source_from_kwargs(_kwargs(trajectory_format="manual-cascade"))
    with pytest.raises(ValueError, match="only shougang"):
        _source_from_kwargs(_kwargs(dataset="finance"))
    with pytest.raises(ValueError, match=r"field_name\+table_name"):
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
