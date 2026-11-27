"""Serialize Skill observations and classification evidence as valid JSON strings."""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path

STAGE = "Execute only the selected Skill INTERPRET_TABLE_NAME."
CLASSIFICATION = {"Execute only the selected Skill INFER_CATEGORY.",
                  "Execute only the selected Skill INFER_CATEGORY_FROM_TOPK."}


def identity():
    return {"version": "skill-json-v1", "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def project(messages):
    stage = messages[0]["content"].splitlines()[0]
    if not stage.startswith("Execute only the selected Skill "):
        return messages
    out = deepcopy(messages)
    if stage in CLASSIFICATION:
        out[0]["content"] += (
            "\nJSON serialization: Write summary and missing_evidence in short plain Chinese sentences. "
            "Do not surround field names or source phrases with ASCII double quotes inside those sentences. "
            "Keep exact citation text in support.citations[].quote. JSON-escape embedded double quotes and "
            "backslashes so that the decoded citation still matches the original source text."
        )
        return out
    if stage != STAGE:
        out[0]["content"] += (
            "\nInside JSON string values write short, plain observations. Avoid adding quotation marks "
            "around names. JSON-escape any required double quotes, backslashes, and line breaks; "
            "preserve the decoded text of exact source quotes and the required output types."
        )
        return out
    payload = json.loads(out[-1]["content"])
    schema = payload["output_schema"]
    if set(schema) != {"summary", "data"} or set(schema["data"]) != {"table_semantics"}:
        raise ValueError("Unexpected native table observation shape")
    schema["summary"] = "one short observation in Chinese"
    schema["data"]["table_semantics"] = "one short Chinese sentence grounded in the supplied table facts"
    out[0]["content"] += (
        "\nWrite one JSON object containing two short string values: summary and data.table_semantics. "
        "Use plain prose inside each value. Avoid enclosing names or descriptions in quotation marks. "
        "If a literal double quote is essential inside a string, JSON-escape it. "
        "Report only the supported table meaning. Where a name is ambiguous, say the meaning is unknown."
    )
    out[-1]["content"] = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return out


@contextmanager
def use_format(runtime, record):
    if record is None:
        yield
        return
    if record != identity():
        raise ValueError("Unexpected table observation protocol")
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
