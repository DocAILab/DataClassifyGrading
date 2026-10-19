"""JSON-compatible contracts for the DSL orchestration layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


OPTIONAL_METADATA_FIELDS = (
    "database_name",
    "database_description",
    "table_name",
    "table_description",
    "field_description",
    "field_type",
    "value",
)


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


@dataclass(frozen=True)
class MetadataRecord:
    """One classification instance with required and optional metadata."""

    domain: str
    field_name: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    source_id: str | None = None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "MetadataRecord":
        nested = raw.get("metadata")
        nested_metadata = dict(nested) if isinstance(nested, Mapping) else {}
        merged = dict(nested_metadata)
        for key in ("field_name", *OPTIONAL_METADATA_FIELDS):
            if key in raw and raw[key] is not None:
                merged[key] = raw[key]

        domain = _text(raw.get("domain", merged.pop("domain", None)))
        field_name = _text(merged.get("field_name"))
        if not domain:
            raise ValueError("domain is required and must be non-empty")
        if not field_name:
            raise ValueError("field_name is required and must be non-empty")
        merged["field_name"] = field_name
        return cls(
            domain=domain,
            field_name=field_name,
            metadata=merged,
            source_id=_text(raw.get("source_id")) or None,
        )

    def get(self, name: str, default: Any = None) -> Any:
        if name == "domain":
            return self.domain
        if name == "field_name":
            return self.field_name
        return self.metadata.get(name, default)

    def available(self, name: str) -> bool:
        value = self.get(name)
        return bool(_text(value))

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"domain": self.domain, "metadata": dict(self.metadata)}
        if self.source_id:
            result["source_id"] = self.source_id
        return result


@dataclass(frozen=True)
class ToolCall:
    tool: str
    arguments: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ToolCall":
        tool = _text(raw.get("tool"))
        if not tool:
            raise ValueError("tool is required and must be non-empty")
        arguments = raw.get("arguments", {})
        if not isinstance(arguments, Mapping):
            raise ValueError("arguments must be an object")
        return cls(tool=tool, arguments=dict(arguments))

    def to_dict(self) -> dict[str, Any]:
        return {"tool": self.tool, "arguments": dict(self.arguments)}


@dataclass(frozen=True)
class Program:
    steps: tuple[ToolCall, ...]

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "Program":
        steps = raw.get("program")
        if not isinstance(steps, list):
            raise ValueError("program must be a list")
        return cls(steps=tuple(ToolCall.from_mapping(item) for item in steps))

    def to_dict(self) -> dict[str, Any]:
        return {"program": [step.to_dict() for step in self.steps]}


@dataclass(frozen=True)
class StandardEntry:
    category: str
    description: str
    grading_level: str

    def to_dict(self) -> dict[str, str]:
        return {
            "category": self.category,
            "description": self.description,
            "grading_level": self.grading_level,
        }

