"""Shared task reward contract (stage 4A): reward_stage1 / reward_stage2."""

from __future__ import annotations

import pytest

from agent.task import LeafRegistry
from agent.training.rl import (
    FULL_REWARD,
    INVALID_REWARD,
    STAGE1_VALID_MISS_DEFAULT,
    STAGE2_PARTIAL_DEFAULT,
    RewardConfig,
    RewardResult,
    reward_stage1,
    reward_stage2,
)


def _registry() -> LeafRegistry:
    return LeafRegistry.from_mapping(["A", "B", "C", "D", "E", "F"])


CANDIDATES = ("A", "B", "C", "D", "E")


def test_stage1_invalid_output_scores_zero() -> None:
    # (output, expected_format_valid)
    cases = (
        ("not json", False),
        ('{"candidates":["A"],"extra":1}', False),  # wrong schema
        ('{"candidates":["A","A","C","D","E"]}', True),  # duplicate
        ('{"candidates":["A","B","Z","D","E"]}', True),  # unknown category
        ('{"candidates":["A","B","C"]}', True),  # wrong count
    )
    for text, expected_format in cases:
        result = reward_stage1(text, ground_truth="C", registry=_registry())
        assert result.reward == pytest.approx(INVALID_REWARD), text
        assert result.format_valid is expected_format, text
        assert result.task_correct is False


def test_stage1_valid_without_ground_truth_scores_stage1_valid_miss() -> None:
    result = reward_stage1(
        '{"candidates":["A","B","F","D","E"]}', ground_truth="C", registry=_registry()
    )
    assert result.reward == pytest.approx(STAGE1_VALID_MISS_DEFAULT)
    assert result.format_valid is True
    assert result.task_correct is False
    assert result.parsed_output is not None
    assert "misses" in result.reason


def test_stage1_valid_with_ground_truth_scores_full() -> None:
    result = reward_stage1(
        '{"candidates":["A","B","C","D","E"]}', ground_truth="C", registry=_registry()
    )
    assert result.reward == pytest.approx(FULL_REWARD)
    assert result.format_valid is True
    assert result.task_correct is True
    assert result.parsed_output is not None


def test_stage1_miss_weight_is_configurable() -> None:
    config = RewardConfig(stage1_valid_miss=0.2)
    result = reward_stage1(
        '{"candidates":["A","B","F","D","E"]}',
        ground_truth="C",
        registry=_registry(),
        config=config,
    )
    assert result.reward == pytest.approx(0.2)


def test_stage2_invalid_output_scores_zero() -> None:
    for text in (
        "garbage",
        '{"candidates":["A","B","C","D","E"]}',
        '{"answer":"F"}',  # outside the candidate bundle
        '{"answer":"C","why":"x"}',
    ):
        result = reward_stage2(
            text, ground_truth="C", candidates=CANDIDATES, registry=_registry()
        )
        assert result.reward == pytest.approx(INVALID_REWARD), text
        assert result.task_correct is False


def test_stage2_valid_but_wrong_scores_partial() -> None:
    result = reward_stage2(
        '{"answer":"B"}', ground_truth="C", candidates=CANDIDATES, registry=_registry()
    )
    assert result.reward == pytest.approx(STAGE2_PARTIAL_DEFAULT)
    assert result.format_valid is True
    assert result.task_correct is False
    assert result.parsed_output is not None
    assert "wrong" in result.reason


def test_stage2_correct_scores_full() -> None:
    result = reward_stage2(
        '{"answer":"C"}', ground_truth="C", candidates=CANDIDATES, registry=_registry()
    )
    assert result.reward == pytest.approx(FULL_REWARD)
    assert result.format_valid is True
    assert result.task_correct is True
    assert result.parsed_output is not None


def test_stage2_partial_is_configurable() -> None:
    config = RewardConfig(stage2_partial=0.25)
    result = reward_stage2(
        '{"answer":"B"}',
        ground_truth="C",
        candidates=CANDIDATES,
        registry=_registry(),
        config=config,
    )
    assert result.reward == pytest.approx(0.25)


def test_reward_result_carries_all_contract_fields() -> None:
    result = reward_stage1(
        '{"candidates":["A","B","C","D","E"]}', ground_truth="C", registry=_registry()
    )
    for field in ("reward", "reason", "parsed_output", "format_valid", "task_correct"):
        assert hasattr(result, field)
    assert isinstance(result, RewardResult)


def test_parser_errors_never_raise_out_of_reward() -> None:
    registry = _registry()
    for text in ("", "{", "[]", '{"candidates": 3}', '{"answer": null}'):
        result = reward_stage1(text, ground_truth="C", registry=registry)
        assert isinstance(result, RewardResult)
        result2 = reward_stage2(
            text, ground_truth="C", candidates=CANDIDATES, registry=registry
        )
        assert isinstance(result2, RewardResult)


def test_programming_errors_still_raise() -> None:
    registry = _registry()
    with pytest.raises(ValueError):
        reward_stage1('{"candidates":["A","B","C","D","E"]}', ground_truth="Z", registry=registry)
    with pytest.raises(ValueError):
        reward_stage2(
            '{"answer":"C"}', ground_truth="C", candidates=("A", "B"), registry=registry
        )
    with pytest.raises(ValueError):
        RewardConfig(stage2_partial=1.5)


def test_reward_config_defaults_are_contract_values() -> None:
    config = RewardConfig()
    assert config.stage1_valid_miss == pytest.approx(0.3)
    assert config.stage2_partial == pytest.approx(0.5)
