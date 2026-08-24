import pytest

from agent.task import (
    PredictionFormatError,
    parse_stage1_output,
    parse_stage2_output,
)


def test_stage1_parser_accepts_only_the_exact_json_shape() -> None:
    parsed = parse_stage1_output('{"candidates":["A","B","C","D","E"]}')

    assert parsed.candidates == ("A", "B", "C", "D", "E")
    with pytest.raises(PredictionFormatError):
        parse_stage1_output(
            '{"candidates":["A","B","C","D","E"],"comment":"extra"}'
        )
    with pytest.raises(PredictionFormatError):
        parse_stage1_output('```json\n{"candidates":["A"]}\n```')


def test_stage2_parser_requires_one_string_answer() -> None:
    parsed = parse_stage2_output('{"answer":"C"}')

    assert parsed.answer == "C"
    with pytest.raises(PredictionFormatError):
        parse_stage2_output('{"answer":3}')
    with pytest.raises(PredictionFormatError):
        parse_stage2_output('{"answer":"C","reason":"extra"}')


def test_parser_rejects_duplicate_json_keys() -> None:
    with pytest.raises(PredictionFormatError, match="duplicate"):
        parse_stage2_output('{"answer":"C","answer":"D"}')
    with pytest.raises(PredictionFormatError, match="duplicate"):
        parse_stage1_output('{"candidates":[],"candidates":[]}')
    with pytest.raises(PredictionFormatError, match="duplicate"):
        parse_stage2_output('{"answer":"1","level":"L1","level":"L2"}', allow_level=True)
