"""Shared, label-free projection for the Bailian teacher classification experiment."""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path

VERSION = "classification-contract-v2"
FINAL = {"INFER_CATEGORY", "INFER_CATEGORY_FROM_TOPK"}
SCOPE_HEADER = "Public definition scope rules (classification policy, not acquired facts):"
DECISION = (
    "Compare the strongest competing definitions before choosing. Identify what the target "
    "field records, whose activity or property it describes, and which acquired fact "
    "distinguishes that scope from the alternative. A table's business theme can supply "
    "context; it cannot by itself establish who owns the target field's activity. "
    "If the same quotes fit competing scopes and no acquired fact distinguishes them, "
    "use the existing UNRESOLVED branch and name the missing distinction. Candidate display "
    "order is by ID. Apply the same test to every option."
)


def identity():
    return {"version": VERSION, "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def by_id(options):
    return sorted(options, key=lambda entry: entry["option_id"])


def project(messages):
    lines = messages[0]["content"].splitlines()
    stage = lines[0].removeprefix("Execute only the selected Skill ").removesuffix(".")
    search = stage.startswith("SEARCH_STANDARD_") or stage == "SELECT_CATEGORY_TOPK"
    if stage not in FINAL and not search:
        return messages
    result = deepcopy(messages)
    if search:
        # Retrieval has a distinct wire schema; the executor constructs standard_candidates.
        if any('{"option_ids":' in line for line in lines):
            lines = [line for line in lines if not line.startswith(
                'Return exactly {"summary": "nonempty natural-language observation", "data":')]
    else:
        if not any("The sole output contract is:" in line for line in lines):
            raise ValueError("Expected the versioned classification output schema")
        lines = [line for line in lines if line not in {
            "# Output", "# Failure", "Return an explicit error when no candidate is supported.",
            "Return JSON with `summary`, `data.category`, and `data.aligned_standard_entry`.",
            "Complete output shapes (structure only; replace every placeholder):",
        } and not line.startswith('[{"option_id":"<one supplied option_id>')]
        payload = json.loads(result[-1]["content"])
        for key in ("candidate_details", "candidate_scope_index"):
            payload[key] = by_id(payload[key])
        if SCOPE_HEADER in lines:
            pos = lines.index(SCOPE_HEADER) + 1
            lines[pos] = json.dumps(by_id(json.loads(lines[pos])), ensure_ascii=False, separators=(",", ":"))
        lines.append(DECISION)
        result[-1]["content"] = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    result[0]["content"] = "\n".join(lines)
    return result


@contextmanager
def use_protocol(runtime, record):
    if record is None:  # Previously collected R14 traces use their original projector.
        yield
        return
    if record != identity():
        raise ValueError("Unknown or modified classification protocol")
    original = runtime.CommonActor

    class SharedActor(original):
        def project(self, messages):
            view = super().project(messages)
            # The original projector already compared the two actor inputs byte for byte.
            shared = project(view["messages"])
            return {**view, "messages": shared, "input_sha256": runtime.previous.message_sha(shared)}

    runtime.CommonActor = SharedActor
    try:
        yield
    finally:
        runtime.CommonActor = original
