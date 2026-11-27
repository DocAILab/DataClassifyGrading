"""Shared lookup for the mentor DSL. Only public metadata locates a table."""
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path


def identity():
    return {"version": "public-table-names-v1", "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def install(controller):
    lookup = controller.bridge.executor.metadata_lookup
    tables = defaultdict(list)
    database_names = defaultdict(set)
    for entry in lookup._rows.values():
        fields = entry["fields"]
        if fields.get("table_name"):
            key = (entry["domain"], fields["table_name"])
            tables[key].append(fields)
            if fields.get("database_name"):
                database_names[key].add(fields["database_name"])

    def neighbors(record):
        own = lookup.get(record)
        key = (record.domain, own.get("table_name"))
        rows = tables.get(key, ())
        if len(database_names.get(key, ())) > 1:
            db = own.get("database_name")
            rows = [r for r in rows if db and r.get("database_name") == db]
        ordered = sorted(rows, key=lambda r: (r["field_name"], json.dumps(r, sort_keys=True, ensure_ascii=False)))
        distinct = {}
        for row in ordered:
            if row["field_name"] != record.field_name:
                distinct.setdefault(row["field_name"], {k: row[k] for k in
                    ("field_name", "field_description", "field_type", "value") if k in row})
        return distinct

    def read_names(record):
        own = lookup.get(record)
        rows = neighbors(record)
        fields = [{"field_name": name} for name in list(rows)[:30]]
        result = dict(database_name=own.get("database_name"), table_name=own.get("table_name"),
                      fields=fields, available_count=len(rows), returned_count=len(fields),
                      omitted_count=len(rows)-len(fields), truncated=len(rows)>len(fields),
                      coverage="canonical-val-same-table-only",
                      missing_locators=[k for k in ("database_name", "table_name") if k not in own])
        if own.get("table_name") and not own.get("database_name"):
            # The native binder checks this spelling and verifies the entire observation.
            result.update(coverage="declared-source-table-only-v1",
                          scope_resolution=dict(status="declared_named_source_table", physical_database_known=False))
        return result

    def expand(record, names):
        if not isinstance(names, list) or not names or not all(isinstance(n, str) for n in names) or len(set(names)) != len(names):
            raise ValueError("Select distinct field names from the acquired table list")
        rows = neighbors(record)
        if not set(names).issubset(list(rows)[:30]):
            raise ValueError("Selected field is outside the visible table window")
        return [deepcopy(rows[n]) for n in names]

    def read_context(record):
        own = lookup.get(record)
        result = read_names(record)
        rows = list(neighbors(record).values())
        keys = ("field_name", "database_name", "database_description", "table_name", "table_description",
                "field_description", "field_type", "value")
        result.update(target=own, fields=deepcopy(rows), returned_count=len(rows), omitted_count=0,
                      truncated=False, missing_target_attributes=[k for k in keys if k not in own])
        return result

    lookup.table_fields = read_names
    lookup.expand = expand
    lookup.table_context = read_context
    return controller
