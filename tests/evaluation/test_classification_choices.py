"""Evaluation adapters: model outputs speak choice ids, evaluation stays canonical."""

from agent.evaluation import (
    evaluate_stage1_choices,
    evaluate_stage2_choices,
)
from agent.task import LeafRegistry, PromptChoiceRegistry


def _registry() -> LeafRegistry:
    return LeafRegistry.from_mapping(["A", "B", "C", "D", "E", "F"])


def _choices() -> PromptChoiceRegistry:
    return PromptChoiceRegistry.from_registry(_registry())


# candidate canonical ids for the fixture below
CANDIDATES = ("C", "A", "B", "D", "E")


def test_stage1_choices_decode_and_recall_ground_truth() -> None:
    # choice ids: A=1 B=2 C=3 D=4 E=5 F=6
    evaluation = evaluate_stage1_choices(
        '{"candidates":["3","1","2","4","5"]}',
        ground_truth="C",
        registry=_registry(),
        choices=_choices(),
    )

    assert evaluation.format_valid is True
    assert evaluation.contract_valid is True
    assert evaluation.ground_truth_recalled is True
    # prediction is the DECODED canonical tuple
    assert evaluation.prediction == ("C", "A", "B", "D", "E")
    assert evaluation.errors == ()


def test_stage1_choices_not_recalled_is_contract_valid() -> None:
    evaluation = evaluate_stage1_choices(
        '{"candidates":["1","2","4","5","6"]}',
        ground_truth="C",
        registry=_registry(),
        choices=_choices(),
    )

    assert evaluation.contract_valid is True
    assert evaluation.ground_truth_recalled is False
    assert evaluation.prediction == ("A", "B", "D", "E", "F")


def test_stage1_choices_reject_unknown_choice_id() -> None:
    evaluation = evaluate_stage1_choices(
        '{"candidates":["3","1","2","4","9"]}',
        ground_truth="C",
        registry=_registry(),
        choices=_choices(),
    )

    assert evaluation.format_valid is True
    assert evaluation.contract_valid is False
    assert evaluation.ground_truth_recalled is False
    assert evaluation.prediction is None
    assert any("not in the prompt catalog" in error for error in evaluation.errors)


def test_stage1_choices_reject_duplicates_and_wrong_count() -> None:
    duplicate = evaluate_stage1_choices(
        '{"candidates":["1","1","2","4","5"]}',
        ground_truth="C",
        registry=_registry(),
        choices=_choices(),
    )
    short = evaluate_stage1_choices(
        '{"candidates":["1","2","3"]}',
        ground_truth="C",
        registry=_registry(),
        choices=_choices(),
    )

    assert duplicate.contract_valid is False
    assert any("unique" in error for error in duplicate.errors)
    assert short.contract_valid is False
    assert any("exactly 5" in error for error in short.errors)


def test_stage1_choices_reject_non_json() -> None:
    evaluation = evaluate_stage1_choices(
        "not json",
        ground_truth="C",
        registry=_registry(),
        choices=_choices(),
    )

    assert evaluation.format_valid is False
    assert evaluation.contract_valid is False
    assert evaluation.prediction is None


def test_stage2_choices_decode_local_id_to_canonical() -> None:
    correct = evaluate_stage2_choices(
        '{"answer":"1"}',
        ground_truth="C",
        candidates=CANDIDATES,
        registry=_registry(),
    )
    wrong = evaluate_stage2_choices(
        '{"answer":"2"}',
        ground_truth="C",
        candidates=CANDIDATES,
        registry=_registry(),
    )

    assert correct.format_valid is True
    assert correct.contract_valid is True
    assert correct.correct is True
    assert correct.prediction == "C"
    assert wrong.contract_valid is True
    assert wrong.correct is False
    assert wrong.prediction == "A"


def test_stage2_choices_reject_invalid_local_ids() -> None:
    outside = evaluate_stage2_choices(
        '{"answer":"6"}',
        ground_truth="C",
        candidates=CANDIDATES,
        registry=_registry(),
    )
    non_numeric = evaluate_stage2_choices(
        '{"answer":"A"}',
        ground_truth="C",
        candidates=CANDIDATES,
        registry=_registry(),
    )

    assert outside.contract_valid is False
    assert outside.correct is False
    assert any("one of 1..5" in error for error in outside.errors)
    assert non_numeric.contract_valid is False


def test_stage2_choices_reject_non_json() -> None:
    evaluation = evaluate_stage2_choices(
        "not json",
        ground_truth="C",
        candidates=CANDIDATES,
        registry=_registry(),
    )

    assert evaluation.format_valid is False
    assert evaluation.contract_valid is False
    assert evaluation.prediction is None
