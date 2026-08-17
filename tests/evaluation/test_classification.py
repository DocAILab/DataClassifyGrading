from agent.evaluation import evaluate_stage1, evaluate_stage2
from agent.task import LeafRegistry


def _registry() -> LeafRegistry:
    return LeafRegistry.from_mapping(["A", "B", "C", "D", "E", "F"])


def test_stage1_evaluation_separates_format_contract_and_recall() -> None:
    valid = evaluate_stage1(
        '{"candidates":["A","B","C","D","E"]}',
        ground_truth="C",
        registry=_registry(),
    )
    duplicate = evaluate_stage1(
        '{"candidates":["A","A","C","D","E"]}',
        ground_truth="C",
        registry=_registry(),
    )
    malformed = evaluate_stage1(
        "not json",
        ground_truth="C",
        registry=_registry(),
    )

    assert valid.format_valid is True
    assert valid.contract_valid is True
    assert valid.ground_truth_recalled is True
    assert valid.prediction == ("A", "B", "C", "D", "E")
    assert duplicate.format_valid is True
    assert duplicate.contract_valid is False
    assert duplicate.ground_truth_recalled is False
    assert malformed.format_valid is False
    assert malformed.contract_valid is False
    assert malformed.prediction is None


def test_stage2_evaluation_separates_membership_and_correctness() -> None:
    candidates = ("A", "B", "C", "D", "E")
    correct = evaluate_stage2(
        '{"answer":"C"}',
        ground_truth="C",
        candidates=candidates,
        registry=_registry(),
    )
    wrong = evaluate_stage2(
        '{"answer":"B"}',
        ground_truth="C",
        candidates=candidates,
        registry=_registry(),
    )
    outside = evaluate_stage2(
        '{"answer":"F"}',
        ground_truth="C",
        candidates=candidates,
        registry=_registry(),
    )

    assert correct.format_valid is True
    assert correct.contract_valid is True
    assert correct.correct is True
    assert correct.prediction == "C"
    assert wrong.contract_valid is True
    assert wrong.correct is False
    assert outside.format_valid is True
    assert outside.contract_valid is False
    assert outside.correct is False
