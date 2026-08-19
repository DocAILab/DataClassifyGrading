"""Choice-protocol reward (Phase 7): reward_stage1/2_choices.

RL rollouts answer with choice ids; the shared choice-aware parser decodes
them to canonical category ids before the unchanged reward table applies.
These tests pin the reward behavior (correct / valid-wrong / invalid /
malformed) and the agreement between evaluation and reward on the validity
of the same output.
"""

from __future__ import annotations

import pytest

from agent.evaluation import evaluate_stage1_choices, evaluate_stage2_choices
from agent.task import LeafRegistry, PromptChoiceRegistry
from agent.training.rl import (
    FULL_REWARD,
    INVALID_REWARD,
    STAGE1_VALID_MISS_DEFAULT,
    STAGE2_PARTIAL_DEFAULT,
    RewardConfig,
    RewardResult,
    reward_stage1_choices,
    reward_stage2_choices,
)


def _registry() -> LeafRegistry:
    return LeafRegistry.from_mapping(["A", "B", "C", "D", "E", "F"])


def _choices() -> PromptChoiceRegistry:
    return PromptChoiceRegistry.from_registry(_registry())


# local ids in candidate order: C=1 A=2 B=3 D=4 E=5 (ground truth = C)
CANDIDATES = ("C", "A", "B", "D", "E")


def test_stage1_choice_correct_scores_full() -> None:
    # choice ids: A=1 B=2 C=3 D=4 E=5 F=6 -> candidates C A B D E recall C
    result = reward_stage1_choices(
        '{"candidates":["3","1","2","4","5"]}',
        ground_truth="C",
        registry=_registry(),
        choices=_choices(),
    )
    assert result.reward == pytest.approx(FULL_REWARD)
    assert result.format_valid is True
    assert result.constraint_valid is True
    assert result.task_correct is True


def test_stage1_choice_valid_but_miss_scores_valid_miss() -> None:
    # A B D E F do not include C: valid output, partial credit
    result = reward_stage1_choices(
        '{"candidates":["1","2","4","5","6"]}',
        ground_truth="C",
        registry=_registry(),
        choices=_choices(),
    )
    assert result.reward == pytest.approx(STAGE1_VALID_MISS_DEFAULT)
    assert result.format_valid is True
    assert result.constraint_valid is True
    assert result.task_correct is False


def test_stage1_choice_invalid_scores_zero() -> None:
    for text in (
        '{"candidates":["3","1","2","4","9"]}',  # unknown choice id
        '{"candidates":["1","1","2","4","5"]}',  # duplicate
        '{"candidates":["1","2","3"]}',  # wrong count
    ):
        result = reward_stage1_choices(
            text, ground_truth="C", registry=_registry(), choices=_choices()
        )
        assert result.reward == pytest.approx(INVALID_REWARD), text
        assert result.format_valid is True, text
        assert result.constraint_valid is False, text
        assert result.task_correct is False


def test_stage2_choice_correct_scores_full() -> None:
    result = reward_stage2_choices(
        '{"answer":"1"}', ground_truth="C", candidates=CANDIDATES, registry=_registry()
    )
    assert result.reward == pytest.approx(FULL_REWARD)
    assert result.format_valid is True
    assert result.constraint_valid is True
    assert result.task_correct is True


def test_stage2_choice_valid_but_wrong_scores_partial() -> None:
    result = reward_stage2_choices(
        '{"answer":"2"}', ground_truth="C", candidates=CANDIDATES, registry=_registry()
    )
    assert result.reward == pytest.approx(STAGE2_PARTIAL_DEFAULT)
    assert result.format_valid is True
    assert result.constraint_valid is True
    assert result.task_correct is False


def test_stage2_choice_invalid_answer_scores_zero() -> None:
    for answer in ("0", "6", "A"):
        result = reward_stage2_choices(
            f'{{"answer":"{answer}"}}',
            ground_truth="C",
            candidates=CANDIDATES,
            registry=_registry(),
        )
        assert result.reward == pytest.approx(INVALID_REWARD), answer
        assert result.format_valid is True, answer
        assert result.constraint_valid is False, answer
        assert result.task_correct is False


def test_malformed_json_scores_zero_without_raising() -> None:
    for text in ("", "{", "[]", "not json", '{"candidates": 3}', '{"answer": null}'):
        result = reward_stage1_choices(
            text, ground_truth="C", registry=_registry()
        )  # default choices path
        assert isinstance(result, RewardResult), text
        assert result.reward == pytest.approx(INVALID_REWARD), text
        assert result.format_valid is False, text
        result2 = reward_stage2_choices(
            text, ground_truth="C", candidates=CANDIDATES, registry=_registry()
        )
        assert isinstance(result2, RewardResult), text
        assert result2.reward == pytest.approx(INVALID_REWARD), text
        assert result2.format_valid is False, text


def test_choice_reward_config_is_unchanged_and_applied() -> None:
    config = RewardConfig(stage1_valid_miss=0.2, stage2_partial=0.25)
    miss = reward_stage1_choices(
        '{"candidates":["1","2","4","5","6"]}',
        ground_truth="C",
        registry=_registry(),
        choices=_choices(),
        config=config,
    )
    wrong = reward_stage2_choices(
        '{"answer":"2"}',
        ground_truth="C",
        candidates=CANDIDATES,
        registry=_registry(),
        config=config,
    )
    assert miss.reward == pytest.approx(0.2)
    assert wrong.reward == pytest.approx(0.25)
    # defaults remain the contract values
    defaults = RewardConfig()
    assert defaults.stage1_valid_miss == pytest.approx(STAGE1_VALID_MISS_DEFAULT)
    assert defaults.stage2_partial == pytest.approx(STAGE2_PARTIAL_DEFAULT)


def test_choice_reward_result_carries_contract_fields() -> None:
    result = reward_stage1_choices(
        '{"candidates":["3","1","2","4","5"]}',
        ground_truth="C",
        registry=_registry(),
        choices=_choices(),
    )
    for field in (
        "reward",
        "reason",
        "parsed_output",
        "format_valid",
        "constraint_valid",
        "task_correct",
    ):
        assert hasattr(result, field)


def test_programming_errors_still_raise_for_choice_reward() -> None:
    with pytest.raises(ValueError):
        reward_stage1_choices(
            '{"candidates":["1","2","3","4","5"]}',
            ground_truth="Z",
            registry=_registry(),
            choices=_choices(),
        )
    with pytest.raises(ValueError):
        reward_stage2_choices(
            '{"answer":"1"}', ground_truth="C", candidates=("C", "A"), registry=_registry()
        )


STAGE1_OUTPUTS = (
    '{"candidates":["3","1","2","4","5"]}',  # valid, ground truth recalled
    '{"candidates":["1","2","4","5","6"]}',  # valid, ground truth missed
    '{"candidates":["3","1","2","4","9"]}',  # unknown choice id
    '{"candidates":["1","1","2","4","5"]}',  # duplicate
    '{"candidates":["1","2","3"]}',  # wrong count
    "not json",
    "{",
    '{"candidates":["1","2",3,"4","5"]}',  # non-string member
)

STAGE2_OUTPUTS = (
    '{"answer":"1"}',  # correct
    '{"answer":"2"}',  # valid wrong
    '{"answer":"0"}',
    '{"answer":"6"}',
    '{"answer":"A"}',
    "garbage",
    "{",
    '{"answer":5}',  # non-string answer
)


@pytest.mark.parametrize("text", STAGE1_OUTPUTS)
def test_evaluation_and_reward_agree_on_stage1_validity(text: str) -> None:
    evaluation = evaluate_stage1_choices(
        text, ground_truth="C", registry=_registry(), choices=_choices()
    )
    reward = reward_stage1_choices(
        text, ground_truth="C", registry=_registry(), choices=_choices()
    )
    assert evaluation.format_valid == reward.format_valid, text
    assert evaluation.contract_valid == reward.constraint_valid, text


@pytest.mark.parametrize("text", STAGE2_OUTPUTS)
def test_evaluation_and_reward_agree_on_stage2_validity(text: str) -> None:
    evaluation = evaluate_stage2_choices(
        text, ground_truth="C", candidates=CANDIDATES, registry=_registry()
    )
    reward = reward_stage2_choices(
        text, ground_truth="C", candidates=CANDIDATES, registry=_registry()
    )
    assert evaluation.format_valid == reward.format_valid, text
    assert evaluation.contract_valid == reward.constraint_valid, text
