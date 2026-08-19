"""Unified output parser contract (stage 4A): check_stage1/2_output."""

from __future__ import annotations

import pytest

from agent.task import (
    LeafRegistry,
    ParseResult,
    check_stage1_output,
    check_stage2_output,
)


def _registry() -> LeafRegistry:
    return LeafRegistry.from_mapping(["A", "B", "C", "D", "E", "F"])


CANDIDATES = ("A", "B", "C", "D", "E")


def test_stage1_invalid_json_is_structured_failure() -> None:
    result = check_stage1_output("not json at all", registry=_registry())
    assert isinstance(result, ParseResult)
    assert result.format_valid is False
    assert result.constraint_valid is False
    assert result.ok is False
    assert result.output is None
    assert result.errors


def test_stage1_wrong_schema_is_structured_failure() -> None:
    for text in (
        '{"candidates":["A","B","C","D","E"],"comment":"extra"}',  # extra key
        '{"answer":"A"}',  # wrong key
        '["A","B","C","D","E"]',  # not an object
        '{"candidates":["A",5,"C","D","E"]}',  # non-string member
        "```json\n{\"candidates\":[\"A\"]}\n```",  # markdown fence
    ):
        result = check_stage1_output(text, registry=_registry())
        assert result.format_valid is False, text
        assert result.ok is False


def test_stage1_duplicate_candidates_fail_constraints() -> None:
    result = check_stage1_output(
        '{"candidates":["A","A","C","D","E"]}', registry=_registry()
    )
    assert result.format_valid is True
    assert result.constraint_valid is False
    assert result.output is not None and result.output.candidates == ("A", "A", "C", "D", "E")
    assert any("unique" in error for error in result.errors)


def test_stage1_unknown_candidate_fails_constraints() -> None:
    result = check_stage1_output(
        '{"candidates":["A","B","Z","D","E"]}', registry=_registry()
    )
    assert result.format_valid is True
    assert result.constraint_valid is False
    assert any("registry" in error for error in result.errors)


def test_stage1_wrong_count_fails_constraints() -> None:
    result = check_stage1_output('{"candidates":["A","B","C"]}', registry=_registry())
    assert result.format_valid is True
    assert result.constraint_valid is False
    assert any("exactly 5" in error for error in result.errors)
    # expected_count is configurable
    two = check_stage1_output(
        '{"candidates":["A","B"]}', registry=_registry(), expected_count=2
    )
    assert two.ok is True


def test_stage1_valid_output_without_ground_truth_passes_constraints() -> None:
    result = check_stage1_output(
        '{"candidates":["A","B","F","D","E"]}', registry=_registry()
    )
    assert result.ok is True
    assert result.format_valid is True
    assert result.constraint_valid is True
    assert result.output is not None
    assert "C" not in result.output.candidates


def test_stage1_correct_output_passes() -> None:
    result = check_stage1_output(
        '{"candidates":["A","B","C","D","E"]}', registry=_registry()
    )
    assert result.ok is True
    assert "C" in result.output.candidates


def test_stage2_invalid_json_and_schema() -> None:
    for text in (
        "garbage",
        '{"candidates":["A"]}',
        '{"answer":"A","reason":"x"}',
        '{"answer":5}',
    ):
        result = check_stage2_output(text, candidates=CANDIDATES)
        assert result.format_valid is False, text
        assert result.ok is False


def test_stage2_answer_outside_candidates_fails_constraints() -> None:
    result = check_stage2_output('{"answer":"F"}', candidates=CANDIDATES)
    assert result.format_valid is True
    assert result.constraint_valid is False
    assert result.output is not None and result.output.answer == "F"


def test_stage2_valid_wrong_answer_passes_constraints() -> None:
    result = check_stage2_output('{"answer":"B"}', candidates=CANDIDATES)
    assert result.ok is True
    assert result.output is not None and result.output.answer == "B"


def test_stage2_correct_answer_passes() -> None:
    result = check_stage2_output('{"answer":"C"}', candidates=CANDIDATES)
    assert result.ok is True


def test_checks_never_raise_on_model_output() -> None:
    registry = _registry()
    garbage = [
        "",
        None,  # type: ignore[arg-type]
        "{",
        "[]",
        '{"candidates": 3}',
        '{"answer": null}',
        "\x00\x01binary",
        "{}",
    ]
    for text in garbage:
        result = check_stage1_output(text, registry=registry)
        assert isinstance(result, ParseResult)
        result2 = check_stage2_output(text, candidates=CANDIDATES)
        assert isinstance(result2, ParseResult)


def test_candidate_contract_mismatch_is_programming_error() -> None:
    with pytest.raises(ValueError):
        check_stage2_output('{"answer":"A"}', candidates="A")  # bare string
    with pytest.raises(ValueError):
        check_stage2_output('{"answer":"A"}', candidates=[])  # empty
