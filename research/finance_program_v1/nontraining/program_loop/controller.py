"""Bounded outer feedback loop; no Planner calls within a Program."""
from collections import Counter
from collections.abc import Mapping
from copy import deepcopy
import json
from pathlib import Path

from .contracts import ProgramConfig, ProgramTrace, json_snapshot
from .catalog_context import skill_library
from .execution import ExecutionBridge
from .planner import planner_messages
from .validation import parse_program
from .rejection_context import rejected_program_summary
from agent.orchestration.contracts import MetadataRecord


def _load(value):
    return json.loads(Path(value).read_text(encoding="utf-8-sig")) if isinstance(value, (str, Path)) else deepcopy(value)


class ProgramController:
    def __init__(self, generate, registry, metadata, config):
        self.generate = generate
        self.config = config
        self.bridge = ExecutionBridge(generate, registry, metadata, config)
        self.library = skill_library(self.bridge.catalog, self.bridge.loader)

    def remaining(self, trace):
        return {"programs": max(0, self.config.max_programs - trace.budget["programs"]),
                "planner_attempts": max(0, self.config.max_planner_attempts - trace.budget["planner_attempts"]),
                "skill_steps": max(0, self.config.max_skill_steps - trace.budget["skill_steps"]),
                "program_steps": self.config.max_program_steps}

    def run(self, record):
        if isinstance(record, MetadataRecord):
            record = {"domain": record.domain, "field_name": record.field_name, "source_id": record.source_id}
        # Ignore caller optional metadata and labels; this boundary exposes only initial fields.
        visible = MetadataRecord.from_mapping({"domain": record.get("domain"), "field_name": record.get("field_name") or record.get("metadata", {}).get("field_name"), "source_id": record.get("source_id")})
        trace = ProgramTrace({"domain": visible.domain, "field_name": visible.field_name})
        state = {}; self.bridge.availability.initialize(visible, state)
        counts = Counter(); lineage = {"evidence": 0, "candidates": 0, "selection": 0}
        feedback = None
        while not trace.terminated:
            remaining = self.remaining(trace)
            exhausted = next((key for key in ("skill_steps", "programs", "planner_attempts") if remaining[key] == 0), None)
            if exhausted:
                trace.stop_reason = {"skill_steps": "skill_budget_exhausted", "programs": "program_budget_exhausted", "planner_attempts": "planner_budget_exhausted"}[exhausted]
                break
            messages = planner_messages(visible, state, self.library, remaining, feedback)
            trace.budget["planner_attempts"] += 1
            attempt = {"attempt": trace.budget["planner_attempts"], "messages": deepcopy(messages), "raw_response": None, "status": "error"}
            trace.planner_attempts.append(attempt)
            try:
                raw = self.generate(messages)
                attempt["raw_response"] = json_snapshot(raw)
                attempt["status"] = "rejected"
                plan = parse_program(raw, self.bridge.catalog, state, self.config.max_program_steps)
            except Exception as exc:
                attempt["error"] = str(exc)
                feedback = {"status": "planner_" + attempt["status"], "error": str(exc), "remaining_budget": self.remaining(trace)}
                if attempt['status'] == 'rejected':
                    summary = rejected_program_summary(raw, self.bridge.catalog, self.config.max_program_steps)
                    if summary is not None:
                        feedback['rejected_program'] = summary
                continue
            attempt["status"] = "accepted"
            trace.budget["programs"] += 1
            observation = {"program_index": trace.budget["programs"], "plan": plan.to_dict(), "steps": [], "status": "completed", "failure_index": None, "terminated": False, "invalidations": []}
            trace.programs.append(observation)
            for index, call in enumerate(plan.steps):
                if self.remaining(trace)["skill_steps"] == 0:
                    observation.update(status="budget_exhausted", failure_index=index)
                    break
                trace.budget["skill_steps"] += 1
                step = self.bridge.step(call, visible, state, counts, lineage)
                step["step_index"] = index
                observation["steps"].append(step)
                observation["invalidations"].extend(step["invalidations"])
                if step["status"] != "success":
                    observation.update(status="failed", failure_index=index)
                    break
                if self.bridge.catalog[call.tool].terminal:
                    trace.terminated = True
                    trace.final_result = deepcopy(step["data"]["final_result"])
                    trace.stop_reason = "validated_terminal"
                    observation.update(status="terminated", terminated=True)
                    break
            observation["remaining_budget"] = self.remaining(trace)
            observation["lineage"] = dict(lineage)
            feedback = deepcopy(observation)
        trace.state = deepcopy(state)
        return trace


def build_program_controller(generate, registry_entries, metadata_rows, config=None):
    if not callable(generate): raise TypeError("generate must be callable")
    registry = _load(registry_entries)
    if isinstance(registry, Mapping) and "categories" in registry:
        registry = {item["category_id"]: {k: v for k, v in item.items() if k not in ("category", "category_id")} for item in registry["categories"]}
    if not isinstance(registry, Mapping) or not registry: raise ValueError("nonempty frozen Registry mapping is required")
    for category, entry in registry.items():
        if not isinstance(category, str) or not category or not isinstance(entry, Mapping) or not isinstance(entry.get("grading_level"), str) or not entry["grading_level"]:
            raise ValueError("Registry entries require category and frozen grading_level")
    config = ProgramConfig(**config) if isinstance(config, Mapping) else config or ProgramConfig()
    return ProgramController(generate, registry, _load(metadata_rows), config)
