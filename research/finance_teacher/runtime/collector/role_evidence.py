"""Highlight literal relationships already present in acquired field descriptions."""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re


def identity():
    return {"version": "related-field-descriptions-v1",
            "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def qualifier(value):
    if not isinstance(value, str):
        return None
    # Only an explicit separator qualifies. Similar column names or code prefixes
    # do not establish a relationship between otherwise unrelated descriptions.
    parts = re.split(r"[-－—:：]", value, maxsplit=1)
    if len(parts) == 2 and parts[0].strip() and parts[1].strip():
        return parts[0].strip()
    return None


def project(messages):
    head = messages[0]["content"].splitlines()[0]
    if not (head in {"Execute only the selected Skill INFER_CATEGORY.",
                     "Execute only the selected Skill INFER_CATEGORY_FROM_TOPK.",
                     "Execute only the selected Skill SELECT_CATEGORY_TOPK."}
            or head.startswith("Execute only the selected Skill SEARCH_STANDARD")):
        return messages
    body = json.loads(messages[-1]["content"])
    facts = body.get("evidence_boundary", {}).get("facts", {})
    target = facts.get("metadata.field_description", {})
    shared = qualifier(target.get("value"))
    if not shared or target.get("role") != "raw_fact" or not target.get("decisive_eligible"):
        return messages
    related = {fid: fact["value"] for fid, fact in facts.items()
               if re.fullmatch(r"observations\.table_fields\.fields\.\d+\.field_description", fid)
               and fact.get("role") == "raw_fact" and fact.get("decisive_eligible") is True
               and qualifier(fact.get("value")) == shared}
    if not related:
        return messages
    body["related_field_descriptions"] = {"shared_literal": shared, "target": target["value"],
                                           "same_table": related}
    result = deepcopy(messages)
    result[-1]["content"] = json.dumps(body, ensure_ascii=False, separators=(",", ":"))
    result[0]["content"] += (
        "\nUse related_field_descriptions to compare the named role of the target with the roles of "
        "other columns in the same row. Determine what one row records, which participant the target "
        "describes, and whether each candidate describes that participant in that activity. "
        "A definition can be supported by the schema without proving who held an individual account "
        "or what value appeared in every row. A shared literal is a reading aid only; it supplies "
        "no extra observation, identity, or ownership. Keep other participants distinct. "
        "When a distinction remains undecidable, retain the existing uncertainty rules.")
    return result


@contextmanager
def use(runtime, record):
    if record is None:
        yield
        return
    if record != identity():
        raise ValueError("Unexpected related-field protocol")
    previous = runtime.CommonActor

    class Actor(previous):
        def project(self, messages):
            view = super().project(messages)
            shared = project(view["messages"])
            return {**view, "messages": shared, "input_sha256": runtime.previous.message_sha(shared)}

    runtime.CommonActor = Actor
    try:
        yield
    finally:
        runtime.CommonActor = previous
