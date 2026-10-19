"""Label-isolated lookup of real field metadata for an explicit DSL call."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .contracts import MetadataRecord


RETRIEVABLE_FIELDS = (
    "database_name", "table_name", "field_description", "table_description"
)


class FieldMetadataLookup:
    def __init__(self, rows: Mapping[str, Mapping[str, Any]]) -> None:
        self._rows = {key: dict(value) for key, value in rows.items()}

    @classmethod
    def from_canonical(cls, source: str | Path) -> "FieldMetadataLookup":
        raw = json.loads(Path(source).read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError("canonical source must be a list")
        projected: dict[str, dict[str, Any]] = {}
        for row in raw:
            if not isinstance(row, Mapping):
                raise ValueError("canonical row must be an object")
            source_id = row.get("id")
            if not isinstance(source_id, str) or not source_id:
                raise ValueError("canonical row must have id")
            if source_id in projected:
                raise ValueError(f"duplicate source id: {source_id}")
            metadata = row.get("metadata")
            if not isinstance(metadata, Mapping):
                raise ValueError(f"canonical metadata missing for {source_id}")
            projected[source_id] = {
                "domain": row.get("domain"),
                "field_name": metadata.get("field_name"),
                "fields": {
                    key: str(metadata[key]).strip()
                    for key in RETRIEVABLE_FIELDS
                    if metadata.get(key) not in (None, "")
                    and str(metadata[key]).strip().casefold() not in ('', 'none', 'null', 'n/a')
                },
            }
        return cls(projected)

    def get(self, record: MetadataRecord) -> dict[str, str]:
        if not record.source_id or record.source_id not in self._rows:
            raise ValueError("source id is absent from metadata lookup")
        entry = self._rows[record.source_id]
        if entry["domain"] != record.domain or entry["field_name"] != record.field_name:
            raise ValueError("lookup source does not match domain and field_name")
        return dict(entry["fields"])
