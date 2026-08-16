"""Strict parsers for model outputs in the two-stage classification task."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any


class PredictionFormatError(ValueError):
    """Raised when a model output does not match the exact JSON shape."""


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
