"""Show the same already-observed selected-column descriptions to teacher and student."""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
from unittest.mock import patch


SELECTED = re.compile(r"observations\.(?:context_fields|context_field_descriptions\.fields)\.\d+\.field_description")
PLANNER_RULE = ("When two plausible definitions differ by whose activity a row records or by the purpose of "
                "that activity, identify the missing distinction. If a table name has been observed, choose a "
                "small relevant context lookup that can distinguish them before deciding. Skip it if the "
                "observed field itself establishes the scope. Reuse facts already acquired.")


def identity():
    return {"version":"selected-field-evidence-v1","sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def project(messages):
    first = messages[0]["content"].splitlines()[0]
    if first.startswith("You are the Global Program Planner"):
        out = deepcopy(messages)
        out[0]["content"] += "\n" + PLANNER_RULE
        return out
    if first not in ("Execute only the selected Skill INFER_CATEGORY.",
                     "Execute only the selected Skill INFER_CATEGORY_FROM_TOPK."):
        return messages
    payload = json.loads(messages[-1]["content"])
    facts = payload.get("evidence_boundary", {}).get("facts", {})
    selected = []
    seen = set()
    for fid, fact in facts.items():
        value = fact.get("value")
        if (SELECTED.fullmatch(fid) and fact.get("role") == "raw_fact"
                and fact.get("source_scope") == "target_or_same_table_context"
                and fact.get("decisive_eligible") is True and isinstance(value, str) and value.strip()
                and value not in seen):
            selected.append({"fact_id":fid, "quote":value})
            seen.add(value)
    if not selected:
        return messages
    payload["selected_field_evidence"] = selected
    out = deepcopy(messages)
    out[-1]["content"] = json.dumps(payload,ensure_ascii=False,separators=(",",":"))
    out[0]["content"] += ("\nselected_field_evidence lists literal descriptions the planner actually acquired. "
                         "Check which participant and activity these columns jointly identify. Decide whether "
                         "the target describes the participant itself or its role in that activity. Use exact "
                         "fact IDs for support and keep genuine unresolved distinctions explicit.")
    return out


@contextmanager
def use(runtime, record):
    if record is None:
        yield
        return
    if record != identity():
        raise ValueError("Unexpected selected-field evidence protocol")
    class Actor(runtime.CommonActor):
        def project(self, native):
            view = super().project(native)
            messages = project(view["messages"])
            return {**view,"messages":messages,"input_sha256":runtime.previous.message_sha(messages)}
    with patch.object(runtime, "CommonActor", Actor):
        yield
