"""Strict compact comparison wire parser; no registry, backend, or state access."""
import json
from collections.abc import Mapping


MAX_REASON_LENGTH = 240


class ComparisonOutputError(ValueError):
    """A model response violates the compact comparison wire contract."""


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ComparisonOutputError(f"duplicate comparison key: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ComparisonOutputError(f"non-JSON comparison constant: {value}")


def parse_comparison_output(raw, candidate_option_ids):
    """Validate one exact JSON decision against the actual current candidate IDs."""
    if not isinstance(raw, str) or not raw.strip():
        raise ComparisonOutputError("comparison response must be nonempty JSON text")
    try:
        value = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    except (json.JSONDecodeError, RecursionError) as exc:
        raise ComparisonOutputError(f"invalid comparison JSON: {exc}") from exc
    if not isinstance(value, Mapping) or set(value) != {"decision", "option_id", "reason"}:
        raise ComparisonOutputError("comparison must contain exactly decision, option_id, reason")
    reason = value["reason"]
    if not isinstance(reason, str) or not reason.strip() or len(reason) > MAX_REASON_LENGTH:
        raise ComparisonOutputError(f"comparison reason must be nonempty text of at most {MAX_REASON_LENGTH} characters")
    decision, option_id = value["decision"], value["option_id"]
    if decision == "match":
        if not isinstance(option_id, str) or option_id not in candidate_option_ids:
            raise ComparisonOutputError("comparison match option_id must be an actual current candidate short ID")
    elif decision == "no_match":
        if option_id is not None:
            raise ComparisonOutputError("comparison no_match option_id must be null")
    else:
        raise ComparisonOutputError("comparison decision must be match or no_match")
    return dict(value)
