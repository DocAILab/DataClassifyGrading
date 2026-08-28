from __future__ import annotations

import pytest

from script.verl.rl.evaluate_native_tool_loop import (
    aggregate_gate,
    normalize_episode_row,
)


def _rl_row() -> dict:
    return {
        "data_source": "shougang/stage1",
        "prompt": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
        ],
        "reward_model": {"style": "rule", "ground_truth": "B1-1-2"},
        "extra_info": {
            "dataset": "shougang",
            "stage": "stage1",
            "source_id": "source-1",
            "metadata": {
                "field_name": "F",
                "table_name": "T",
                "field_description": "D",
                "table_description": "TD",
            },
            "ground_truth_level": "L2",
            "trajectory_format": "qwen3.5-native-tools-v2",
        },
    }


def test_normalize_rl_row_keeps_exact_agent_loop_contract() -> None:
    episode = normalize_episode_row(_rl_row())
    assert episode.source_id == "source-1"
    assert episode.raw_prompt[0]["role"] == "system"
    assert episode.extra_info["stage"] == "stage1"
    assert episode.reward_model == {"style": "rule", "ground_truth": "B1-1-2"}


def test_normalize_trajectory_row_uses_only_system_and_user_messages() -> None:
    row = {
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
            {"role": "assistant", "content": "must not be sent"},
        ],
        "stage": "tool_trajectory",
        "trajectory_format": "qwen3.5-native-tools-v2",
        "source_id": "source-2",
        "metadata": {
            "field_name": "F",
            "table_name": "T",
            "field_description": "D",
            "table_description": "TD",
        },
        "ground_truth": "B1-1-2",
        "ground_truth_level": "L3",
    }
    episode = normalize_episode_row(row)
    assert len(episode.raw_prompt) == 2
    assert episode.extra_info["stage"] == "stage1"
    assert episode.extra_info["ground_truth_level"] == "L3"


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda row: row["extra_info"].update({"unknown": 1}), "unexpected keys"),
        (lambda row: row["extra_info"].update({"stage": "stage2"}), "stage1"),
        (lambda row: row["prompt"].append({"role": "assistant", "content": "x"}), "must be exactly system"),
    ],
)
def test_normalize_row_fails_closed(mutate, message: str) -> None:
    row = _rl_row()
    mutate(row)
    with pytest.raises(ValueError, match=message):
        normalize_episode_row(row)


def test_gate_bounds_are_inclusive() -> None:
    assert aggregate_gate([1] + [0] * 19)["passed"] is True
    assert aggregate_gate([1] * 3 + [0] * 7)["passed"] is True
    assert aggregate_gate([0] * 19)["passed"] is False
    assert aggregate_gate([1] * 4 + [0] * 77)["passed"] is False
