"""Strict parsers for model outputs in the two-stage classification task.

Three layers, sharing the exact JSON-shape parsers:

- ``parse_stage1_output`` / ``parse_stage2_output`` raise
  ``PredictionFormatError`` on malformed JSON/schema (used by SFT
  validation);
- ``check_stage1_output`` / ``check_stage2_output`` never raise on model
  output: they return a structured ``ParseResult`` separating format
  validity (JSON + schema) from constraint validity (candidate count /
  uniqueness / registry membership, or answer membership). RL rewards and
  the evaluator consume this layer so malformed model output can never
  crash a training loop.
- ``check_stage1_choices`` / ``check_stage2_choices`` are the SHARED
  choice-aware layer: they run the same JSON/schema parsing, validate the
  choice protocol (stage1: exact count, uniqueness, known choice ids;
  stage2: local id 1..5) and decode to canonical category ids, returning a
  ``ChoiceParseResult``. Both the evaluation adapters and the RL reward
  entry points consume this layer, so choice validation is implemented
  exactly once and the model's choice ids never leak into canonical
  correctness logic.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Sequence

from .contracts import LeafRegistry
from .prompt_choices import PromptChoiceRegistry


class PredictionFormatError(ValueError):
    """Raised when a model output does not match the exact JSON shape."""


@dataclass(frozen=True)
class ParseResult:
    """Structured parse outcome for one model output; never raised.

    - format_valid: the output is a JSON object with exactly the required
      schema (stage1: only ``candidates``; stage2: only ``answer``).
    - constraint_valid: the parsed values satisfy the task constraints
      (stage1: exactly ``expected_count`` unique candidates from the
      registry; stage2: the answer belongs to the candidate list).
    - output: the parsed shape (when format_valid), so consumers never
      need to re-parse.
    - errors: human-readable failure reasons; empty when both valid.
    """

    format_valid: bool
    constraint_valid: bool
    output: "Stage1Output | Stage2Output | None" = None
    errors: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.format_valid and self.constraint_valid

    def __post_init__(self) -> None:
        if not isinstance(self.format_valid, bool) or not isinstance(
            self.constraint_valid, bool
        ):
            raise ValueError("ParseResult validity flags must be bool")
        if self.errors and self.format_valid and self.constraint_valid:
            raise ValueError("a fully valid ParseResult must carry no errors")
        if not isinstance(self.errors, tuple) or not all(
            isinstance(error, str) for error in self.errors
        ):
            raise ValueError("ParseResult errors must be a tuple of strings")


@dataclass(frozen=True)
class Stage1Output:
    candidates: tuple[str, ...]


@dataclass(frozen=True)
class Stage2Output:
    answer: str


def _object(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, TypeError) as exc:
        raise PredictionFormatError("prediction must be a JSON object") from exc
    if not isinstance(value, dict):
        raise PredictionFormatError("prediction must be a JSON object")
    return value


def parse_stage1_output(text: str) -> Stage1Output:
    """Parse the Stage 1 JSON shape without applying registry constraints."""
    value = _object(text)
    if set(value) != {"candidates"}:
        raise PredictionFormatError("stage1 prediction must contain only candidates")
    candidates = value["candidates"]
    if not isinstance(candidates, list) or not all(
        isinstance(candidate, str) for candidate in candidates
    ):
        raise PredictionFormatError("stage1 candidates must be an array of strings")
    return Stage1Output(tuple(candidates))


def parse_stage2_output(text: str) -> Stage2Output:
    """Parse the Stage 2 JSON shape without applying candidate constraints."""
    value = _object(text)
    if set(value) != {"answer"}:
        raise PredictionFormatError("stage2 prediction must contain only answer")
    answer = value["answer"]
    if not isinstance(answer, str):
        raise PredictionFormatError("stage2 answer must be a string")
    return Stage2Output(answer)


def _check_candidates(
    candidates: tuple[str, ...],
    registry: LeafRegistry,
    expected_count: int,
) -> tuple[bool, tuple[str, ...]]:
    """Stage 1 constraint checks: exact count, no duplicates, all in registry."""
    errors: list[str] = []
    if len(candidates) != expected_count:
        errors.append(
            f"stage1 prediction must contain exactly {expected_count} candidates"
        )
    if len(set(candidates)) != len(candidates):
        errors.append("stage1 candidates must be unique")
    if any(candidate not in registry.ids for candidate in candidates):
        errors.append("stage1 candidates must belong to the leaf registry")
    return (not errors, tuple(errors))


def check_stage1_output(
    text: str,
    *,
    registry: LeafRegistry,
    expected_count: int = 5,
) -> ParseResult:
    """Unified Stage 1 parser + contract validation; never raises on model output.

    Validates, in order: JSON object, exact schema (only ``candidates``),
    candidate count, uniqueness, and registry membership. The parsed
    candidates are kept on the result even when constraints fail so callers
    can inspect what the model actually produced.
    """
    if expected_count < 1:
        raise ValueError("expected_count must be positive")
    try:
        output = parse_stage1_output(text)
    except PredictionFormatError as exc:
        return ParseResult(False, False, None, (str(exc),))
    valid, errors = _check_candidates(output.candidates, registry, expected_count)
    return ParseResult(True, valid, output, errors)


def check_stage2_output(text: str, *, candidates: Sequence[str]) -> ParseResult:
    """Unified Stage 2 parser + contract validation; never raises on model output.

    Validates: JSON object, exact schema (only ``answer``), and answer
    membership in the candidate list. Candidates originate from the dataset
    (not the model), so an invalid candidate list is a programming error
    and raises.
    """
    if isinstance(candidates, (str, bytes)) or not candidates:
        raise ValueError("stage2 candidates must be a non-empty sequence of strings")
    candidate_ids = tuple(candidates)
    if not all(isinstance(candidate, str) and candidate for candidate in candidate_ids):
        raise ValueError("stage2 candidates must be non-empty strings")
    try:
        output = parse_stage2_output(text)
    except PredictionFormatError as exc:
        return ParseResult(False, False, None, (str(exc),))
    valid = output.answer in candidate_ids
    errors = () if valid else ("stage2 answer must be one of the candidates",)
    return ParseResult(True, valid, output, errors)


_STAGE2_LOCAL_IDS = tuple(str(index) for index in range(1, 6))


@dataclass(frozen=True)
class ChoiceParseResult:
    """Structured never-raised outcome of the choice-aware parse + decode.

    The choice protocol (``agent.task.prompt_choices``) maps model-facing
    choice ids to canonical category ids. This result mirrors
    ``ParseResult``'s shape contract and additionally carries ``decoded``:
    the canonical category ids the model's choice ids map to (stage1: the
    candidate tuple; stage2: the single answer id). ``output`` always keeps
    the model-level parsed shape (choice ids) when format_valid, so
    consumers can see exactly what the model said before decoding.

    - format_valid: the output is a JSON object with exactly the required
      schema (identical semantics to ParseResult.format_valid).
    - constraint_valid: the parsed choice ids satisfy the choice protocol
      (stage1: exactly ``expected_count`` unique known choice ids; stage2:
      answer is one of the local ids 1..5).
    - decoded: the canonical category ids when constraint_valid (stage1:
      tuple of candidate ids; stage2: the single answer id), else None.
    - errors: human-readable failure reasons; empty when both valid.
    """

    format_valid: bool
    constraint_valid: bool
    decoded: tuple[str, ...] | str | None = None
    output: "Stage1Output | Stage2Output | None" = None
    errors: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.format_valid and self.constraint_valid

    def __post_init__(self) -> None:
        if not isinstance(self.format_valid, bool) or not isinstance(
            self.constraint_valid, bool
        ):
            raise ValueError("ChoiceParseResult validity flags must be bool")
        if self.errors and self.format_valid and self.constraint_valid:
            raise ValueError("a fully valid ChoiceParseResult must carry no errors")
        if not isinstance(self.errors, tuple) or not all(
            isinstance(error, str) for error in self.errors
        ):
            raise ValueError("ChoiceParseResult errors must be a tuple of strings")
        if self.constraint_valid and self.decoded is None:
            raise ValueError(
                "a constraint-valid ChoiceParseResult must carry decoded canonical ids"
            )

    def canonical_view(self) -> ParseResult:
        """Equivalent canonical ``ParseResult`` for the shared eval/reward cores.

        When constraints hold the output is the decoded canonical shape
        (Stage1Output of decoded ids / Stage2Output of the decoded answer);
        when only the format held, the model-level shape is kept so
        consumers can still inspect what the model said.
        """
        if not self.format_valid:
            return ParseResult(False, False, None, self.errors)
        if self.constraint_valid:
            output: Stage1Output | Stage2Output | None = (
                Stage1Output(self.decoded)
                if isinstance(self.decoded, tuple)
                else Stage2Output(self.decoded)
            )
        else:
            output = self.output
        return ParseResult(True, self.constraint_valid, output, self.errors)


def check_stage1_choices(
    text: str,
    *,
    choices: PromptChoiceRegistry,
    expected_count: int = 5,
) -> ChoiceParseResult:
    """Shared choice-aware Stage 1 parser + decode; never raises on model output.

    Validates, in order: JSON object, exact schema (only ``candidates``),
    candidate count, uniqueness, and choice-id membership in the prompt
    catalog; then decodes the choice ids to canonical category ids.
    Malformed model output maps to a structured ``ChoiceParseResult``; only
    a missing/invalid ``choices`` mapping is a programming error and
    raises.
    """
    if expected_count < 1:
        raise ValueError("expected_count must be positive")
    if not isinstance(choices, PromptChoiceRegistry):
        raise ValueError("choices must be a PromptChoiceRegistry")
    try:
        output = parse_stage1_output(text)
    except PredictionFormatError as exc:
        return ChoiceParseResult(False, False, None, None, (str(exc),))
    errors: list[str] = []
    if len(output.candidates) != expected_count:
        errors.append(
            f"stage1 prediction must contain exactly {expected_count} candidates"
        )
    if len(set(output.candidates)) != len(output.candidates):
        errors.append("stage1 candidates must be unique")
    for choice_id in output.candidates:
        if not choices.contains_choice_id(choice_id):
            errors.append(f"choice id {choice_id!r} is not in the prompt catalog")
    if errors:
        return ChoiceParseResult(True, False, None, output, tuple(errors))
    return ChoiceParseResult(
        True,
        True,
        tuple(choices.category_id_of(choice_id) for choice_id in output.candidates),
        output,
        (),
    )


def check_stage2_choices(text: str, *, candidates: Sequence[str]) -> ChoiceParseResult:
    """Shared choice-aware Stage 2 parser + decode; never raises on model output.

    Validates: JSON object, exact schema (only ``answer``), and answer
    membership in the local ids 1..5; then decodes the local id to the
    canonical category id of the candidate at that position. Candidates
    originate from the dataset, so an invalid candidate list is a
    programming error and raises (mirrors check_stage2_output).
    """
    if isinstance(candidates, (str, bytes)) or len(candidates) != 5:
        raise ValueError("stage2 requires exactly 5 candidates for local-id decode")
    candidate_ids = tuple(candidates)
    if not all(isinstance(candidate, str) and candidate for candidate in candidate_ids):
        raise ValueError("stage2 candidates must be non-empty strings")
    try:
        output = parse_stage2_output(text)
    except PredictionFormatError as exc:
        return ChoiceParseResult(False, False, None, None, (str(exc),))
    if output.answer not in _STAGE2_LOCAL_IDS:
        return ChoiceParseResult(
            True,
            False,
            None,
            output,
            (f"stage2 answer {output.answer!r} must be one of 1..5",),
        )
    return ChoiceParseResult(
        True, True, candidate_ids[int(output.answer) - 1], output, ()
    )
