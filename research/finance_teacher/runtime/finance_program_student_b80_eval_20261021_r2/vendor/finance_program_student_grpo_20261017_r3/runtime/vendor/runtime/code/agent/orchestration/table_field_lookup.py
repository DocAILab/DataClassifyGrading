"""Label-isolated, bounded lookup of neighboring fields in one real table."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Mapping

from .contracts import MetadataRecord


TableKey = tuple[str, str, str]


class TableFieldLookup:
    def __init__(
        self,
        identities: Mapping[str, tuple[str, str, TableKey]],
        tables: Mapping[TableKey, list[tuple[str, str]]],
        *,
        max_fields: int = 30,
    ) -> None:
        if max_fields < 1:
            raise ValueError("max_fields must be positive")
        self.max_fields = max_fields
        self._identities = dict(identities)
        self._tables = {key: tuple(values) for key, values in tables.items()}

    @classmethod
    def from_canonical(cls, source: str | Path, *, max_fields: int = 30) -> "TableFieldLookup":
        raw = json.loads(Path(source).read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError("canonical source must be a list")
        identities: dict[str, tuple[str, str, TableKey]] = {}
        tables: dict[TableKey, list[tuple[str, str]]] = defaultdict(list)
        for row in raw:
            if not isinstance(row, Mapping):
                raise ValueError("canonical row must be an object")
            source_id = row.get("id")
            if not isinstance(source_id, str) or not source_id.strip():
                raise ValueError("canonical row must have source id")
            if source_id in identities:
                raise ValueError(f"duplicate source id: {source_id}")
            metadata = row.get("metadata")
            if not isinstance(metadata, Mapping):
                raise ValueError(f"canonical metadata missing for {source_id}")
            domain = str(row.get("domain") or "").strip()
            database = str(metadata.get("database_name") or "").strip()
            table = str(metadata.get("table_name") or "").strip()
            field_name = str(metadata.get("field_name") or "").strip()
            if not domain or not database or not table or not field_name:
                raise ValueError(f"table identity is incomplete for {source_id}")
            key = (domain, database, table)
            identities[source_id] = (domain, field_name, key)
            description = str(metadata.get("field_description") or "").strip()
            tables[key].append((field_name, description))
        return cls(identities, tables, max_fields=max_fields)

    def get(self, record: MetadataRecord) -> dict[str, Any]:
        if not record.source_id or record.source_id not in self._identities:
            raise ValueError("source id is absent from table field lookup")
        domain, field_name, key = self._identities[record.source_id]
        if domain != record.domain or field_name != record.field_name:
            raise ValueError("lookup source does not match domain and field_name")
        selected: dict[str, str] = {}
        for name, description in self._tables[key]:
            if name == record.field_name:
                continue
            previous = selected.get(name, "")
            if description and (not previous or description < previous):
                selected[name] = description
            else:
                selected.setdefault(name, previous)
        names = sorted(selected, key=lambda value: (value.casefold(), value))
        fields: list[dict[str, str]] = []
        for name in names[: self.max_fields]:
            item = {"field_name": name}
            if selected[name]:
                item["field_description"] = selected[name]
            fields.append(item)
        return {
            "fields": fields,
            "available_count": len(names),
            "truncated": len(names) > self.max_fields,
        }


__all__ = ["TableFieldLookup"]
