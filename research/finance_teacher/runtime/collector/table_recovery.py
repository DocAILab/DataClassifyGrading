"""Tell the Planner what another scoped table read can add after an actual gap."""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from unittest.mock import patch


def identity():
    return {"version": "known-table-coverage-v1", "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def project(messages, native):
    if not messages[0]["content"].startswith("You are the Global Program Planner for a database-field classification task."):
        return messages
    state = json.loads(native[-1]["content"])
    if not state.get("evidence_gap_ledger"):
        return messages
    facts = state["evidence_boundary"]["facts"]
    context = facts.get("observations.table_context", {}).get("value", {})
    first = facts.get("observations.table_fields", {}).get("value", {})
    if context and context.get("truncated") is False:
        advice = ("All available columns from this table have already been returned. Re-reading the table "
                  "or rephrasing its columns adds no independent facts. If no remaining available source "
                  "can answer the stated gap, submit the current classification Skill with the existing "
                  "candidates now so the native evidence guard can close the episode.")
    elif first.get("truncated") and first.get("omitted_count", 0) > 0:
        advice = (f"The first table read returned {first['returned_count']} of {first['available_count']} columns. "
                  f"{first['omitted_count']} scoped columns are still unread. To check the gap against these "
                  "columns, use GET_COMPLETE_TABLE_CONTEXT followed by a classification Skill. "
                  "SELECT_CONTEXT_FIELDS and EXPAND_SELECTED_CONTEXT only revisit the first window.")
    else:
        return messages
    out = deepcopy(messages)
    out[0]["content"] += "\nTable coverage for this unresolved field: " + advice
    return out


@contextmanager
def use(runtime, record):
    if record is None:
        yield
        return
    if record != identity():
        raise ValueError("Unknown table recovery protocol")

    class Actor(runtime.CommonActor):
        def project(self, native):
            view = super().project(native)
            shared = project(view["messages"], native)
            return {**view, "messages": shared, "input_sha256": runtime.previous.message_sha(shared)}

    with patch.object(runtime, "CommonActor", Actor):
        yield
