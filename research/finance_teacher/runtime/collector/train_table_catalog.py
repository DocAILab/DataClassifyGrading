"""Optionally complete sparse, named training tables from the paired TRAIN archive."""
from collections import defaultdict
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import source_evidence


def identity():
    return {"version": "named-train-tables-v1", "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def extend(bundle, present):
    boundary = source_evidence.read(source_evidence.DATA / "data/split_boundaries.json")
    raw = (source_evidence.DATA / "raw/DCG_FIN_TRAIN.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != boundary["split_sha256"]["train"]:
        raise ValueError("TRAIN archive differs from the frozen split")
    tables = boundary["tables"]
    held_out = {tuple(pair) for split in ("val", "test") for pair in tables[split]}
    held_ids = set(boundary["ids"]["val"]) | set(boundary["ids"]["test"])
    allowed = set(boundary["ids"]["train"]) - held_ids
    partition = source_evidence.read(source_evidence.SCHEDULE)["source_partition"]
    dev = {i for i, p in partition.items() if p["split"] != "train"}
    source = source_evidence.read(source_evidence.DATA / "raw/DCG_FIN_SFT.json")
    dev_groups = {r["source_table_id"] for r in source if r["id"] in dev}
    dev_tables = {r["source_table_name"] for r in source if r["id"] in dev}

    def table(row):
        return (row["domain"], row["source_table_id"], row["source_database_name"], row["source_table_name"])

    rows = list(bundle["lookup_rows"])
    scopes = list(bundle["lookup_scopes"])
    seen = {r["id"] for r in rows}
    names = defaultdict(set)
    for row in scopes:
        names[table(row)].add(row["metadata"]["field_name"])
    targets = {c["source_id"] for c in bundle["cases"] if c["split"] == "train"}
    sparse = {table(r) for r in scopes if r["id"] in targets and present(r["source_table_name"])
              and r["source_table_id"] not in dev_groups and r["source_table_name"] not in dev_tables
              and (r["source_database_name"], r["source_table_name"]) not in held_out
              and len(names[table(r)]) <= source_evidence.LIMIT}
    # Import original records without annotations. The native reader still selects
    # its usual bounded, sorted window per target, and reports the total coverage.
    columns = ("domain", "source_table_id", "source_database_name", "source_table_name")
    for record in sorted(json.loads(raw), key=lambda r: r["id"]):
        key, sid = table(record), record["id"]
        field = record["metadata"]["field_name"]
        if (key not in sparse or sid not in allowed or sid in seen or sid in dev
                or record["label_status"] != "original" or field in names[key]):
            continue
        public = {"id": sid, "domain": record["domain"], "metadata": record["metadata"]}
        rows.append(public)
        scopes.append({**public, **{k: record[k] for k in columns}})
        names[key].add(field)
        seen.add(sid)
    return {**bundle, "lookup_rows": rows, "lookup_scopes": scopes}


@contextmanager
def use(record):
    if record is None:
        yield
        return
    if record != identity():
        raise ValueError("Unknown training-table catalog")
    original = source_evidence.extend
    with patch.object(source_evidence, "extend", lambda base, present: extend(original(base, present), present)):
        yield
