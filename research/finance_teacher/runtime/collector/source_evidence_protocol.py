"""Choose the next evidence-acquisition step from observed table availability."""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import source_evidence


def identity():
    return {"version": "observe-before-classifying-v2",
            "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "source_metadata": source_evidence.identity()}


def project(messages, native):
    if messages[0]["content"].splitlines()[0] != "You are the Global Program Planner for a database-field classification task.":
        return messages
    payload = json.loads(native[-1]["content"])
    boundary = payload["evidence_boundary"]
    facts = boundary["facts"]
    observed = payload["current"]["named_state"]
    if "field_metadata" not in observed and set(facts) == {"metadata.field_name"}:
        instruction = ("Only the field identifier is visible and its table availability is unknown. "
                       "For this decision return a one-step acquisition program using GET_FIELD_METADATA. "
                       "End that program there so the next planning decision sees the actual metadata.")
    elif ("metadata.table_name" in facts and "table_fields" not in observed
          and not any(item.get("skill") == "GET_TABLE_FIELDS" for item in boundary["availability_ledger"])):
        # A native table read may legitimately return zero rows. Do not loop on it.
        instruction = ("A real table name has been observed. Put GET_TABLE_FIELDS at the beginning of "
                       "the next program, before interpreting the identifier or deciding among candidate "
                       "categories. You may plan the downstream Skills in the same program. Use the roles "
                       "and activities in that table to interpret what the target field records.")
    elif "field_metadata" in observed and "metadata.table_name" not in facts:
        instruction = ("The field metadata has already been read and it supplied no table name. "
                       "Skip table-dependent Skills. Use the observed field description to retrieve and "
                       "compare candidate definitions. If it has several plausible meanings, state exactly "
                       "what is missing and honor an existing insufficient-evidence decision once no "
                       "independent business fact can be acquired.")
    else:
        return messages
    out = deepcopy(messages)
    out[0]["content"] += "\nPlan from acquired facts. " + instruction
    return out


@contextmanager
def use(science, record):
    if record is None or record == source_evidence.identity():
        with source_evidence.use(science, record):
            yield
        return
    if record != identity():
        raise ValueError("Unrecognized source-evidence protocol")
    with source_evidence.use(science, record["source_metadata"]):
        runtime = science.runtime()
        previous = runtime.CommonActor

        class Actor(previous):
            def project(self, native):
                view = super().project(native)
                shared = project(view["messages"], native)
                return {**view, "messages": shared, "input_sha256": runtime.previous.message_sha(shared)}

        runtime.CommonActor = Actor
        try:
            yield
        finally:
            runtime.CommonActor = previous
