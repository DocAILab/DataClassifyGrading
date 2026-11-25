"""Static validation for planner-produced Programs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .catalog import SkillSpec
from .contracts import MetadataRecord, Program


DEFAULT_MAX_STEPS = 10


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str
    step_index: int | None = None

    def __str__(self) -> str:
        location = "" if self.step_index is None else f" at step {self.step_index}"
        return f"{self.code}{location}: {self.message}"


def resolve_reference(value: Any, record: MetadataRecord, state: Mapping[str, Any]) -> tuple[bool, Any]:
    if not isinstance(value, str) or not value.startswith("$"):
        return bool(value is not None and value != ""), value
    if value == "$record":
        # The controller may retain source_id privately for deterministic lookups.
        # It is never included in resolved arguments sent to model-backed Skills.
        return True, {"domain": record.domain, "metadata": dict(record.metadata)}
    if value == "$domain":
        return True, record.domain
    if value.startswith("$metadata."):
        name = value[len("$metadata.") :]
        resolved = record.get(name)
        return record.available(name), resolved
    if value.startswith("$state."):
        name = value[len("$state.") :]
        return name in state, state.get(name)
    return False, None


def validate_program(
    record: MetadataRecord,
    program: Program,
    catalog: Mapping[str, SkillSpec],
    *,
    initial_state: Mapping[str, Any] | None = None,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    state = dict(initial_state or {})
    seen: dict[str, int] = {}
    finalized = False

    if len(program.steps) > max_steps:
        issues.append(ValidationIssue("max_steps_exceeded", f"program has {len(program.steps)} steps; max is {max_steps}"))

    for index, call in enumerate(program.steps):
        spec = catalog.get(call.tool)
        if spec is None:
            issues.append(ValidationIssue("unknown_skill", f"unknown Skill {call.tool!r}", index))
            continue
        if call.tool == "INFER_CATEGORY_FROM_TOPK":
            if not seen.get("SELECT_CATEGORY_TOPK") and "standard_candidates" not in (initial_state or {}):
                issues.append(ValidationIssue(
                    "missing_topk_dependency",
                    "INFER_CATEGORY_FROM_TOPK requires a preceding SELECT_CATEGORY_TOPK",
                    index,
                ))
            if call.arguments.get("candidate_standard_categories") != "$state.standard_candidates":
                issues.append(ValidationIssue(
                    "invalid_topk_reference",
                    "candidate_standard_categories must reference $state.standard_candidates",
                    index,
                ))
        if finalized:
            issues.append(ValidationIssue("calls_after_finalizer", "no calls are allowed after FINALIZE_CLASSIFICATION", index))

        seen[call.tool] = seen.get(call.tool, 0) + 1
        if seen[call.tool] > spec.max_calls:
            issues.append(ValidationIssue("duplicate_skill", f"{call.tool} exceeds max_calls={spec.max_calls}", index))

        for name in spec.required_inputs:
            if name not in call.arguments:
                issues.append(ValidationIssue("missing_required_argument", f"{call.tool} requires argument {name!r}", index))
                continue
            found, _ = resolve_reference(call.arguments[name], record, state)
            if not found:
                value = call.arguments[name]
                if isinstance(value, str) and value.startswith("$metadata."):
                    code = "unavailable_metadata"
                elif isinstance(value, str) and value.startswith("$state."):
                    code = "unresolved_reference"
                else:
                    code = "empty_required_argument"
                issues.append(ValidationIssue(code, f"required argument {name!r} cannot be resolved", index))

        for name in spec.optional_inputs:
            if name in call.arguments:
                resolve_reference(call.arguments[name], record, state)

        if spec.terminal:
            finalized = True
            if index != len(program.steps) - 1:
                issues.append(ValidationIssue("finalizer_not_last", "FINALIZE_CLASSIFICATION must be the last step", index))

        for produced in spec.produces:
            state.setdefault(produced, {"source_skill": call.tool})

    if not finalized:
        issues.append(ValidationIssue("missing_finalizer", "program must end with FINALIZE_CLASSIFICATION"))
    return issues


def assert_valid_program(
    record: MetadataRecord,
    program: Program,
    catalog: Mapping[str, SkillSpec],
    *,
    initial_state: Mapping[str, Any] | None = None,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> None:
    issues = validate_program(record, program, catalog, initial_state=initial_state, max_steps=max_steps)
    if issues:
        raise ValueError("; ".join(str(issue) for issue in issues))


__all__ = ["DEFAULT_MAX_STEPS", "ValidationIssue", "assert_valid_program", "resolve_reference", "validate_program"]
