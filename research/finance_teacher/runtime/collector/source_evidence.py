"""Use original training metadata to fill unused slots in named source tables."""
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "finance_sft_v2_20261001"
SCHEDULE = ROOT.parent / "finance_teacher_same_information_20261024_r1/schedule.json"
LIMIT = 30  # Existing native table_fields limit, excluding the target field.


def identity():
    return {"version": "training-table-evidence-v1", "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def read(path):
    return json.loads(path.read_bytes())


def extend(contract, present):
    boundary = read(DATA / "data/split_boundaries.json")
    excluded = set(boundary["ids"]["val"]) | set(boundary["ids"]["test"])
    dev = {i for i, record in read(SCHEDULE)["source_partition"].items() if record["split"] != "train"}
    # The enclosing R14 freeze authenticates the raw archive and its label-free metadata projection.
    origin = {r["id"]: r for r in read(DATA / "raw/DCG_FIN_SFT.json")}
    blocked_groups = {origin[i]["source_table_id"] for i in dev if i in origin}
    blocked_tables = {origin[i]["source_table_name"] for i in dev if i in origin}
    allowed = {c["source_id"] for c in contract["cases"] if c["split"] == "train"}
    def table(scope):
        return (scope["domain"], scope["source_table_id"], scope["source_database_name"], scope["source_table_name"])
    targets = {table(s) for s in contract["lookup_scopes"]
               if s["id"] in allowed and present(s["source_table_name"])
               and s["source_table_id"] not in blocked_groups and s["source_table_name"] not in blocked_tables}
    rows = list(contract["lookup_rows"])
    scopes = list(contract["lookup_scopes"])
    existing = {r["id"] for r in rows}
    names = defaultdict(set)
    for s in scopes:
        names[table(s)].add(s["metadata"]["field_name"])
    public = read(DATA / "data/metadata-sft.json")
    for row in sorted(public, key=lambda r: r["id"]):
        sid = row["id"]
        if sid in existing or sid in excluded or sid in dev:
            continue
        src = origin[sid]
        group = table(src)
        name = row["metadata"]["field_name"]
        if (group not in targets or src["label_status"] != "original" or name in names[group]
                or len(names[group]) >= LIMIT + 1):
            continue
        if set(row) != {"id", "domain", "metadata"} or row["metadata"] != src["metadata"] or row["domain"] != src["domain"]:
            raise ValueError("Training-row metadata identity mismatch")
        rows.append(row)
        scopes.append({**row, **{k: src[k] for k in ("source_table_id", "source_table_name", "source_database_name")}})
        names[group].add(name)
        existing.add(sid)
    return {**contract, "lookup_rows": rows, "lookup_scopes": scopes}


def project(messages, native):
    head = messages[0]["content"].splitlines()[0]
    is_planner = head == "You are the Global Program Planner for a database-field classification task."
    classify = head in {"Execute only the selected Skill INFER_CATEGORY.", "Execute only the selected Skill INFER_CATEGORY_FROM_TOPK."}
    if not is_planner and not classify:
        return messages
    out = deepcopy(messages)
    if classify:
        out[0]["content"] += (
            "\nRead the target field and verified same-table descriptions together. A combination of distinct "
            "raw observations can establish a business role, a relationship and an activity even if no one field "
            "states a complete category definition. Compare that combination with the closest competing definition. "
            "Cite the distinguishing observed facts by their exact IDs and quotes. Duplicate descriptions, aliases "
            "and repeated observations supply no additional support. Leave the choice unresolved when several "
            "definitions still explain the available facts equally well."
        )
        return out
    state = json.loads(native[-1]["content"])
    body = json.loads(out[-1]["content"])
    facts = state.get("evidence_boundary", {}).get("facts", {})
    missing = state.get("evidence_boundary", {}).get("missing_metadata", [])
    unavailable = {r["metadata_type"] for r in missing if r.get("status") == "unavailable"}
    feedback = body.get("recovery_feedback_v1")
    if feedback:
        feedback["metadata_acquisition_choices"] = [k for k in feedback["metadata_acquisition_choices"] if k not in unavailable]
        feedback["source_confirmed_unavailable"] = sorted(unavailable)
    if "metadata.table_name" in facts and not any(k.startswith("observations.table_fields.") for k in facts):
        advice = ("The acquired target has a named table. When the subject or activity is ambiguous, "
                  "use GET_TABLE_FIELDS before committing to a category; compare several related descriptions.")
    else:
        advice = ("Use acquired same-table descriptions when present. A missing table name cannot establish "
                  "table membership; a field-name prefix or an unrelated table is not a substitute.")
    out[0]["content"] += ("\nEvidence acquisition: " + advice + " Avoid querying attributes already observed as unavailable. "
        "After an insufficient inference, acquire an available fact only if it could resolve the reported distinction. "
        "If no such source remains, submit the same inference Skill with its existing candidates so the native "
        "no-new-evidence guard can close the episode. Never submit an empty program or invent a prediction.")
    out[-1]["content"] = json.dumps(body, ensure_ascii=False, separators=(",", ":"))
    return out


@contextmanager
def use(science, record):
    if record is None:
        yield
        return
    if record != identity():
        raise ValueError("Source-evidence protocol differs from the recorded version")
    rt = science.runtime()
    original_actor, original_build = rt.CommonActor, science.native200.build

    class Actor(original_actor):
        def project(self, native):
            view = super().project(native)
            shared = project(view["messages"], native)
            return {**view, "messages": shared, "input_sha256": rt.previous.message_sha(shared)}

    def build(engine, scoped, stopped, contract):
        return original_build(engine, scoped, stopped, extend(contract, rt.coverage_pool.scope.native.present))

    rt.CommonActor, science.native200.build = Actor, build
    try:
        yield
    finally:
        rt.CommonActor, science.native200.build = original_actor, original_build
