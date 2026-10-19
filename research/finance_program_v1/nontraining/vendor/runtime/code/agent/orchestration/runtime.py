"""CPU-only deterministic indexes and a mock Skill executor."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import re
from typing import Any, Mapping, Sequence

from .catalog import SkillSpec, default_catalog
from .contracts import MetadataRecord, Program, StandardEntry, ToolCall
from .validator import assert_valid_program, resolve_reference


@dataclass
class KnowledgeBase:
    records: tuple[MetadataRecord, ...]
    standards: tuple[StandardEntry, ...]
    _table_index: dict[tuple[str | None, str], list[MetadataRecord]] = field(default_factory=dict)

    @classmethod
    def from_mappings(cls, records: Sequence[Mapping[str, Any]], standards: Sequence[StandardEntry]) -> "KnowledgeBase":
        normalized = tuple(MetadataRecord.from_mapping(record) for record in records)
        index: dict[tuple[str | None, str], list[MetadataRecord]] = defaultdict(list)
        for record in normalized:
            table = record.get("table_name")
            if table:
                index[(record.get("database_name"), str(table))].append(record)
        return cls(normalized, tuple(standards), dict(index))

    def table_fields(self, table_name: str, database_name: str | None = None) -> list[str]:
        rows = self._table_index.get((database_name, table_name), [])
        if not rows and database_name is not None:
            rows = [record for (db, table), values in self._table_index.items() if table == table_name for record in values]
        return list(dict.fromkeys(record.field_name for record in rows))

    def search_standard(self, query: Any, top_k: int = 5) -> list[dict[str, str]]:
        tokens = _search_tokens(query)
        scored = []
        for entry in self.standards:
            haystack_tokens = _search_tokens(f"{entry.category} {entry.description}")
            score = len(tokens.intersection(haystack_tokens))
            scored.append((score, entry))
        scored.sort(key=lambda item: (-item[0], item[1].category))
        return [entry.to_dict() for _, entry in scored[:top_k]]

    def level_for(self, category: str) -> str | None:
        for entry in self.standards:
            if entry.category == category:
                return entry.grading_level
        return None


def _flatten_text(value: Any) -> list[str]:
    if isinstance(value, Mapping):
        parts: list[str] = []
        for key, nested in value.items():
            parts.extend(_flatten_text(key))
            parts.extend(_flatten_text(nested))
        return parts
    if isinstance(value, (list, tuple, set)):
        parts = []
        for nested in value:
            parts.extend(_flatten_text(nested))
        return parts
    return [str(value)] if value is not None else []


def _search_tokens(value: Any) -> set[str]:
    """Extract identifier tokens and CJK n-grams from nested Skill inputs."""

    text = " ".join(_flatten_text(value)).lower()
    tokens = set(re.findall(r"[a-z0-9]+", text))
    for chunk in re.findall(r"[\u4e00-\u9fff]+", text):
        if len(chunk) <= 3:
            tokens.add(chunk)
        tokens.update(chunk[index : index + 2] for index in range(len(chunk) - 1))
        tokens.update(chunk[index : index + 3] for index in range(len(chunk) - 2))
    return {token for token in tokens if token}


@dataclass(frozen=True)
class Observation:
    skill: str
    status: str
    summary: str
    data: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionTrace:
    observations: list[Observation | Mapping[str, Any]] = field(default_factory=list)
    terminated: bool = False
    final_result: dict[str, Any] | None = None
    state: dict[str, Any] = field(default_factory=dict)


class MockSkillExecutor:
    """Run deterministic and placeholder semantic Skills without an LLM."""

    def __init__(self, knowledge_base: KnowledgeBase, catalog: Mapping[str, SkillSpec] | None = None):
        self.knowledge_base = knowledge_base
        self.catalog = dict(catalog or default_catalog())

    def execute(self, call: ToolCall, record: MetadataRecord, state: Mapping[str, Any]) -> Observation:
        spec = self.catalog[call.tool]
        resolved = {
            key: resolve_reference(value, record, state)[1]
            for key, value in call.arguments.items()
        }

        if call.tool == "CHECK_METADATA_AVAILABILITY":
            available = ["domain", "field_name"] + [name for name in record.metadata if record.available(name) and name != "field_name"]
            missing = [name for name in ("database_name", "database_description", "table_name", "table_description", "field_description", "field_type", "value") if not record.available(name)]
            data = {"metadata_availability": {"available": available, "missing": missing}}
        elif call.tool == "GET_TABLE_FIELDS":
            table_name = str(resolved["table_name"])
            data = {"table_fields": self.knowledge_base.table_fields(table_name, record.get("database_name"))}
        elif call.tool == "SELECT_CONTEXT_FIELDS":
            fields = list(resolved["table_fields"] or [])
            target = str(resolved["target_field"])
            data = {"selected_context_fields": [field for field in fields if field != target][:3]}
        elif call.tool == "SEARCH_STANDARD":
            data = {"standard_candidates": self.knowledge_base.search_standard(resolved["semantic_query"])}
        elif call.tool == "INFER_CATEGORY":
            candidates = resolved["candidate_standard_categories"]
            if isinstance(candidates, Mapping):
                candidates = [candidates]
            candidates = list(candidates or [])
            first = dict(candidates[0]) if candidates else {}
            category = first.get("category") or first.get("category_id") or first.get("name")
            data = {"category": category, "aligned_standard_entry": first}
        elif call.tool == "INFER_GRADING_LEVEL":
            entry = resolved["aligned_standard_entry"]
            if isinstance(entry, Mapping):
                level = entry.get("grading_level")
            else:
                level = self.knowledge_base.level_for(str(resolved["aligned_category"]))
            data = {"grading_level": level}
        elif call.tool == "FINALIZE_CLASSIFICATION":
            data = {"final_result": {"category": state.get("category"), "grading_level": state.get("grading_level")}}
        else:
            produced = spec.produces[0] if spec.produces else "observation"
            data = {produced: {"input": resolved, "source_skill": call.tool}}

        return Observation(call.tool, "success", f"Mock observation from {call.tool}", data)


def execute_program(
    record: MetadataRecord,
    program: Program,
    executor: MockSkillExecutor,
    *,
    initial_state: Mapping[str, Any] | None = None,
) -> ExecutionTrace:
    assert_valid_program(record, program, executor.catalog, initial_state=initial_state)
    state = dict(initial_state or {})
    trace = ExecutionTrace(state=state)
    for call in program.steps:
        observation = executor.execute(call, record, state)
        trace.observations.append(observation)
        if observation.status != "success":
            break
        state.update(observation.data)
        if call.tool == "FINALIZE_CLASSIFICATION":
            trace.terminated = True
            trace.final_result = dict(observation.data["final_result"])
            break
    return trace


__all__ = ["ExecutionTrace", "KnowledgeBase", "MockSkillExecutor", "Observation", "execute_program"]
