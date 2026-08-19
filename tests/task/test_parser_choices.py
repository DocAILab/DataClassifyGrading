"""Shared choice-aware parser layer (Phase 7): check_stage1/2_choices.

Model outputs speak choice ids; this layer validates the choice protocol
and decodes to canonical category ids in ONE shared implementation that
both evaluation adapters and RL reward entry points consume.
"""

from __future__ import annotations

import pytest

from agent.task import (
    ChoiceParseResult,
    LeafRegistry,
    ParseResult,
    PromptChoiceRegistry,
    check_stage1_choices,
    check_stage2_choices,
)


def _registry() -> LeafRegistry:
    return LeafRegistry.from_mapping(["A", "B", "C", "D", "E", "F"])


def _choices() -> PromptChoiceRegistry:
    return PromptChoiceRegistry.from_registry(_registry())


# canonical candidates for stage2 fixtures; local ids: C=1 A=2 B=3 D=4 E=5
CANDIDATES = ("C", "A", "B", "D", "E")


def test_stage1_valid_choice_decodes_to_canonical_ids() -> None:
    # choice ids: A=1 B=2 C=3 D=4 E=5 F=6
    result = check_stage1_choices('{"candidates":["3","1","2","4","5"]}', choices=_choices())
    assert isinstance(result, ChoiceParseResult)
    assert result.ok is True
    assert result.format_valid is True
    assert result.constraint_valid is True
    assert result.decoded == ("C", "A", "B", "D", "E")
    assert result.errors == ()
    # the model-level shape is preserved for inspection
    assert result.output is not None
    assert result.output.candidates == ("3", "1", "2", "4", "5")


def test_stage1_unknown_choice_id_fails_constraints() -> None:
    result = check_stage1_choices('{"candidates":["3","1","2","4","9"]}', choices=_choices())
    assert result.format_valid is True
    assert result.constraint_valid is False
    assert result.ok is False
    assert result.decoded is None
    assert any("9" in error and "prompt catalog" in error for error in result.errors)


def test_stage1_duplicate_and_wrong_count_fail_constraints() -> None:
    duplicate = check_stage1_choices(
        '{"candidates":["1","1","2","4","5"]}', choices=_choices()
    )
    short = check_stage1_choices('{"candidates":["1","2","3"]}', choices=_choices())
    long = check_stage1_choices(
        '{"candidates":["1","2","3","4","5","6"]}', choices=_choices()
    )
    assert duplicate.constraint_valid is False
    assert any("unique" in error for error in duplicate.errors)
    assert short.constraint_valid is False
    assert long.constraint_valid is False
    assert any("exactly 5" in error for error in short.errors)
    assert any("exactly 5" in error for error in long.errors)
    # expected_count is configurable, like check_stage1_output
    two = check_stage1_choices('{"candidates":["1","2"]}', choices=_choices(), expected_count=2)
    assert two.ok is True
    assert two.decoded == ("A", "B")


def test_stage1_malformed_json_is_format_failure() -> None:
    for text in (
        "not json",
        "{",
        "[]",
        '{"candidates":["1","2",3,"4","5"]}',  # non-string member
        '{"candidates":["1","2","3","4","5"],"extra":1}',  # extra key
        '{"answer":"1"}',  # wrong key
    ):
        result = check_stage1_choices(text, choices=_choices())
        assert result.format_valid is False, text
        assert result.ok is False
        assert result.decoded is None


def test_stage1_valid_choice_without_ground_truth_still_decodes() -> None:
    result = check_stage1_choices('{"candidates":["1","2","4","5","6"]}', choices=_choices())
    assert result.ok is True
    assert result.decoded == ("A", "B", "D", "E", "F")


def test_stage2_valid_local_answer_decodes_positionally() -> None:
    result = check_stage2_choices('{"answer":"1"}', candidates=CANDIDATES)
    assert result.ok is True
    assert result.decoded == "C"
    assert result.output is not None and result.output.answer == "1"
    assert check_stage2_choices('{"answer":"5"}', candidates=CANDIDATES).decoded == "E"


def test_stage2_answer_outside_local_ids_fails_constraints() -> None:
    for answer in ("0", "6", "A", "01"):
        result = check_stage2_choices(f'{{"answer":"{answer}"}}', candidates=CANDIDATES)
        assert result.format_valid is True, answer
        assert result.constraint_valid is False, answer
        assert result.decoded is None, answer
        assert any("one of 1..5" in error for error in result.errors), answer


def test_stage2_malformed_json_is_format_failure() -> None:
    for text in (
        "garbage",
        "{",
        '{"answer":5}',  # non-string answer
        '{"candidates":["C","A","B","D","E"]}',  # wrong key
        '{"answer":"1","why":"x"}',  # extra key
    ):
        result = check_stage2_choices(text, candidates=CANDIDATES)
        assert result.format_valid is False, text
        assert result.ok is False


def test_choice_checks_never_raise_on_model_output() -> None:
    garbage = [
        "",
        None,  # type: ignore[arg-type]
        "{",
        "[]",
        '{"candidates": 3}',
        '{"answer": null}',
        "{}",
        "\x00\x01binary",
    ]
    for text in garbage:
        result = check_stage1_choices(text, choices=_choices())  # type: ignore[arg-type]
        assert isinstance(result, ChoiceParseResult)
        result2 = check_stage2_choices(text, candidates=CANDIDATES)  # type: ignore[arg-type]
        assert isinstance(result2, ChoiceParseResult)


def test_canonical_view_matches_parse_result_semantics() -> None:
    # constraint-valid -> decoded canonical shape
    view = check_stage1_choices(
        '{"candidates":["3","1","2","4","5"]}', choices=_choices()
    ).canonical_view()
    assert isinstance(view, ParseResult)
    assert view.ok is True
    assert view.output is not None and view.output.candidates == ("C", "A", "B", "D", "E")
    # format-valid but constraint-invalid -> model-level shape kept
    bad = check_stage1_choices(
        '{"candidates":["3","1","2","4","9"]}', choices=_choices()
    ).canonical_view()
    assert bad.format_valid is True and bad.constraint_valid is False
    assert bad.output is not None and bad.output.candidates == ("3", "1", "2", "4", "9")
    # format-invalid -> None output
    junk = check_stage1_choices("nope", choices=_choices()).canonical_view()
    assert junk.format_valid is False and junk.output is None
    # stage2 canonical view carries the decoded answer
    view2 = check_stage2_choices('{"answer":"2"}', candidates=CANDIDATES).canonical_view()
    assert view2.ok is True
    assert view2.output is not None and view2.output.answer == "A"


def test_programming_errors_still_raise() -> None:
    with pytest.raises(ValueError):
        check_stage1_choices(
            '{"candidates":["1","2","3","4","5"]}', choices=None  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError):
        check_stage1_choices('{"candidates":["1","2"]}', choices=_choices(), expected_count=0)
    with pytest.raises(ValueError):
        check_stage2_choices('{"answer":"1"}', candidates=("C", "A"))  # not 5
    with pytest.raises(ValueError):
        check_stage2_choices('{"answer":"1"}', candidates="C")  # bare string
