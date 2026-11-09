"""Current acquired-input facts only; conditional producers are never guaranteed."""

from pool_metadata import FIELDS
from pool_runtime import acquired_raw


def input_readiness(record, state: dict, input_rules: dict) -> dict:
    raw = acquired_raw(record, state)
    metadata = {}
    for name in FIELDS:
        status = "observed" if name in raw else "unknown"
        if name not in raw and any(
            row.get("scope") == "record"
            and row.get("target") == record.field_name
            and row.get("type") == name
            and row.get("status") == "unavailable"
            for row in state.get("availability_ledger", [])
        ):
            status = "observed_unavailable"
        metadata[name] = {
            "status": status,
            "possible_producers": list(
                input_rules.get(name, {}).get("possible_producers", [])
            ),
        }
    return {
        "metadata_inputs": metadata,
        "new_business_facts": False,
        "rule": "Unknown is unqueried, not global absence. Unavailable is an actual source-local observation. Possible producers may omit optional outputs: inspect the real observation before using required metadata. This summary supplies no values or plan, adds no evidence, and does not change binding guards.",
    }
