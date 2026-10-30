"""Run and evaluate CPU-scripted dynamic Agent Loops without loading a model."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import tempfile
from typing import Any, Callable, Iterable, Mapping, Sequence

from agent.orchestration.agent_loop import (
    AgentLoopController,
    FixedPipelinePolicy,
    LoopConfig,
    LoopEvent,
    LoopTrace,
    PrefetchNeighborFieldsPolicy,
    PrefetchFieldMetadataPolicy,
    ScriptedLoopPolicy,
    ScriptedSkillExecutor,
)
from agent.orchestration.catalog import dynamic_loop_catalog
from agent.orchestration.contracts import MetadataRecord, StandardEntry
from agent.orchestration.model_skill_executor import (
    ModelLoopPolicy,
    ModelSkillExecutor,
    TextGenerator,
    load_few_shot_messages,
    make_standard_search_handler,
    make_field_metadata_handler,
    make_table_fields_handler,
)
from agent.orchestration.metadata_lookup import FieldMetadataLookup
from agent.orchestration.table_field_lookup import TableFieldLookup
from agent.orchestration.category_catalog import CategoryCatalog
from agent.orchestration.artifact_projection import project_trace
from agent.orchestration.runtime import KnowledgeBase
from agent.orchestration.runtime import Observation
from agent.orchestration.skill_loader import SkillLoader


ControllerFactory = Callable[[Mapping[str, Any]], AgentLoopController]


@dataclass(frozen=True)
class ArtifactPaths:
    manifest: Path
    traces: Path
    summary: Path
    observations: Path | None = None
    model_calls: Path | None = None
    trajectory: Path | None = None
    evaluation: Path | None = None


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    raise ValueError(f"row {line_number} in {path} must be an object")
                rows.append(dict(value))
        return rows
    value = _read_json(path)
    if not isinstance(value, list) or any(not isinstance(row, Mapping) for row in value):
        raise ValueError(f"{path} must contain a JSON list of objects")
    return [dict(row) for row in value]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_entry(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def _registry_categories(value: Any) -> list[str]:
    if isinstance(value, Mapping):
        entries = value.get("categories")
    else:
        entries = value
    if not isinstance(entries, list):
        raise ValueError("registry must be a list or contain a categories list")
    names: list[str] = []
    for entry in entries:
        if isinstance(entry, str):
            name = entry
        elif isinstance(entry, Mapping):
            name = str(
                entry.get("category_id")
                or entry.get("category")
                or entry.get("name")
                or entry.get("level_4")
                or ""
            )
        else:
            name = ""
        if name:
            names.append(name)
    return list(dict.fromkeys(names))


def _registry_entry_map(value: Any) -> dict[str, dict[str, Any]]:
    entries = value.get("categories") if isinstance(value, Mapping) else value
    if not isinstance(entries, list):
        raise ValueError("registry must be a list or contain a categories list")
    result: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if isinstance(entry, str):
            normalized = {"category": entry, "description": entry, "grading_level": ""}
            category = entry
        elif isinstance(entry, Mapping):
            category = str(
                entry.get("category_id")
                or entry.get("category")
                or entry.get("name")
                or entry.get("level_4")
                or ""
            )
            normalized = dict(entry)
            normalized["category"] = category
        else:
            continue
        if category:
            if category in result:
                raise ValueError(f"duplicate registry category {category!r}")
            result[category] = normalized
    return result


def select_pilot_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    count: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Select a stable pilot independent of input ordering and predictions."""

    if count < 1:
        raise ValueError("count must be positive")
    normalized = [dict(row) for row in rows]
    if any(not row.get("source_id") for row in normalized):
        raise ValueError("every pilot row must have source_id")
    normalized.sort(key=lambda row: str(row["source_id"]))
    source_ids = [str(row["source_id"]) for row in normalized]
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("pilot source_id values must be unique")
    if count > len(normalized):
        raise ValueError(f"requested {count} pilot rows from only {len(normalized)} rows")
    rng = random.Random(seed)
    rng.shuffle(normalized)
    return normalized[:count]


def pending_rows_for_resume(
    rows: Iterable[Mapping[str, Any]],
    traces_path: str | Path,
) -> list[dict[str, Any]]:
    """Return only rows whose source IDs are absent from an existing trace file."""

    path = Path(traces_path)
    completed: set[str] = set()
    if path.is_file():
        for row in _read_rows(path):
            source_id = row.get("source_id")
            if source_id is not None:
                completed.add(str(source_id))
    return [
        dict(row)
        for row in rows
        if str(row.get("source_id")) not in completed
    ]


def _trace_from_dict(value: Mapping[str, Any]) -> LoopTrace:
    return LoopTrace(
        input=dict(value.get("input") or {}),
        events=[
            LoopEvent(
                kind=str(event.get("kind")),
                turn=int(event.get("turn") or 0),
                data=dict(event.get("data") or {}),
            )
            for event in value.get("events", [])
            if isinstance(event, Mapping)
        ],
        terminated=bool(value.get("terminated")),
        final_result=(
            dict(value["final_result"])
            if isinstance(value.get("final_result"), Mapping)
            else None
        ),
        stop_reason=str(value.get("stop_reason") or "unknown"),
        turns=int(value.get("turns") or 0),
        state=dict(value.get("state") or {}),
    )


def _recalculate_saved_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Recompute a resumed summary from preserved traces without model calls."""

    replay_rows: list[dict[str, Any]] = []
    traces: dict[str, LoopTrace] = {}
    for index, raw in enumerate(rows):
        trace_value = raw.get("trace")
        if not isinstance(trace_value, Mapping):
            raise ValueError("saved row is missing its trace object")
        source_id = str(raw.get("source_id") or f"saved-{index}")
        trace = _trace_from_dict(trace_value)
        traces[source_id] = trace
        replay_rows.append(
            {
                "source_id": source_id,
                "domain": trace.input.get("domain"),
                "field_name": trace.input.get("field_name"),
                "ground_truth": raw.get("ground_truth"),
                "ground_truth_level": raw.get("ground_truth_level"),
            }
        )

    class ReplayController:
        def __init__(self, trace: LoopTrace) -> None:
            self.trace = trace

        def run(self, _record: MetadataRecord) -> LoopTrace:
            return self.trace

    return evaluate_agent_rows(
        replay_rows,
        lambda row: ReplayController(traces[str(row["source_id"])]),
    )


def load_experiment_variants(path: str | Path) -> list[dict[str, Any]]:
    """Expand and validate the checked-in A/B/C experiment contract."""

    payload = _read_json(Path(path))
    if not isinstance(payload, Mapping):
        raise ValueError("experiment config must be an object")
    common = payload.get("common")
    raw_variants = payload.get("variants")
    if not isinstance(common, Mapping) or not isinstance(raw_variants, list):
        raise ValueError("experiment config requires common and variants")
    variants: list[dict[str, Any]] = []
    for raw in raw_variants:
        if not isinstance(raw, Mapping):
            raise ValueError("each experiment variant must be an object")
        variant = dict(common)
        variant.update(dict(raw))
        variants.append(variant)
    if [item.get("id") for item in variants] != ["A", "B", "C"]:
        raise ValueError("variants must be ordered A, B, C")
    if variants[0].get("policy") != "fixed" or any(
        variants[index].get("policy") != "dynamic" for index in (1, 2)
    ):
        raise ValueError("A must be fixed and B/C must be dynamic")
    if variants[1].get("few_shot") is not None or not variants[2].get("few_shot"):
        raise ValueError("B must omit few-shot and C must enable it")
    return variants


def build_model_controller(
    *,
    generate: TextGenerator,
    policy_mode: str,
    skill_root: str | Path,
    registry_entries: Mapping[str, Mapping[str, Any]],
    config: LoopConfig | None = None,
    few_shot_messages: Sequence[Mapping[str, str]] = (),
    top_k: int = 5,
    metadata_lookup: FieldMetadataLookup | None = None,
    table_field_lookup: TableFieldLookup | None = None,
    neighbor_prefetch: bool = False,
    metadata_prefetch: bool = False,
    include_description_skills: bool = False,
    topk_evidence_mode: str = "augmented",
    topk_selection_mode: str = "single",
    topk_selection_seed: int = 20260930,
    top1_evidence_mode: str = "all",
    top1_selection_mode: str = "model",
    classification_mode: str = "legacy",
) -> AgentLoopController:
    """Assemble one model-backed controller without importing Transformers."""

    if classification_mode not in {"legacy", "two-stage"}:
        raise ValueError("classification_mode must be legacy or two-stage")
    if classification_mode == "two-stage" and policy_mode == "fixed":
        raise ValueError("two-stage mode currently requires the dynamic policy")
    if neighbor_prefetch and table_field_lookup is None:
        raise ValueError("neighbor_prefetch requires table_field_lookup")
    if neighbor_prefetch and policy_mode != "dynamic":
        raise ValueError("neighbor_prefetch requires the dynamic policy")
    if metadata_prefetch and (
        policy_mode != "dynamic" or classification_mode != "two-stage"
        or metadata_lookup is None or not include_description_skills
    ):
        raise ValueError("metadata_prefetch requires dynamic two-stage classification, metadata_lookup and description Skills")
    if include_description_skills and metadata_lookup is None:
        raise ValueError("description Skills require metadata_lookup")
    if include_description_skills and (classification_mode != "two-stage" or policy_mode != "dynamic"):
        raise ValueError("description Skills require dynamic two-stage classification")
    if topk_selection_mode not in {"single", "balanced-vote"}:
        raise ValueError("topk_selection_mode must be single or balanced-vote")
    if topk_selection_mode == "balanced-vote" and (classification_mode != "two-stage" or policy_mode != "dynamic"):
        raise ValueError("balanced-vote requires dynamic two-stage classification")
    if topk_evidence_mode not in {"augmented", "raw", "raw-flat"}:
        raise ValueError("topk_evidence_mode must be augmented, raw or raw-flat")
    if topk_evidence_mode in {"raw", "raw-flat"} and not include_description_skills:
        raise ValueError("raw Top-K evidence mode requires description Skills")
    two_stage = classification_mode == "two-stage"
    catalog = dynamic_loop_catalog(
        include_field_metadata=metadata_lookup is not None,
        include_table_fields=table_field_lookup is not None,
        include_description_skills=include_description_skills,
        two_stage=two_stage,
    )
    loader = SkillLoader(
        skill_root, catalog,
        extra_roots=(
            *((Path(skill_root) / "metadata_extra",) if metadata_lookup is not None else ()),
            *((Path(skill_root) / "neighbor_extra",) if table_field_lookup is not None else ()),
            *((Path(skill_root) / "semantic_extra",) if include_description_skills else ()),
            *((Path(skill_root) / "two_stage",) if two_stage else ()),
        ),
        strict_catalog=not two_stage,
    )
    standards = tuple(
        StandardEntry(
            category=str(category),
            description=str(entry.get("description") or entry.get("definition") or category),
            grading_level=str(entry.get("grading_level") or ""),
        )
        for category, entry in registry_entries.items()
    )
    knowledge_base = KnowledgeBase.from_mappings([], standards)
    handlers = {"SEARCH_STANDARD": make_standard_search_handler(knowledge_base, top_k=top_k)}
    if metadata_lookup is not None:
        handlers["GET_FIELD_METADATA"] = make_field_metadata_handler(metadata_lookup)
    if table_field_lookup is not None:
        handlers["GET_TABLE_FIELDS"] = make_table_fields_handler(table_field_lookup)
    executor = ModelSkillExecutor(
        generate,
        loader,
        deterministic_handlers=handlers,
        category_catalog=CategoryCatalog(registry_entries) if two_stage else None,
        top_k=top_k,
        guard_two_stage_evidence=include_description_skills,
        topk_evidence_mode=topk_evidence_mode,
        topk_selection_mode=topk_selection_mode,
        topk_selection_seed=topk_selection_seed,
        top1_evidence_mode=top1_evidence_mode,
        top1_selection_mode=top1_selection_mode,
    )
    if policy_mode == "fixed":
        policy = FixedPipelinePolicy()
    elif policy_mode == "dynamic":
        policy = ModelLoopPolicy(
            generate,
            few_shot_messages=few_shot_messages,
            metadata_guidance=metadata_lookup is not None,
            two_stage=two_stage,
            description_skills=include_description_skills,
        )
        if neighbor_prefetch:
            policy = PrefetchNeighborFieldsPolicy(policy)
        if metadata_prefetch:
            policy = PrefetchFieldMetadataPolicy(policy)
    else:
        raise ValueError("policy_mode must be fixed or dynamic")
    return AgentLoopController(
        policy=policy,
        executor=executor,
        catalog=catalog,
        skill_loader=loader,
        config=config or LoopConfig(),
        registry_entries=registry_entries,
    )


def build_run_manifest(
    *,
    dataset_id: str,
    input_path: str | Path,
    registry_path: str | Path,
    corpus_path: str | Path | None,
    code_revision: str,
    sample_count: int,
    seed: int,
    behavior_files: Mapping[str, str | Path] | None = None,
    model: Mapping[str, Any] | None = None,
    loop_config: Mapping[str, Any] | None = None,
    working_tree_dirty: bool | None = None,
) -> dict[str, Any]:
    """Build an auditable CPU-run manifest from exact files."""

    input_file = Path(input_path)
    registry_file = Path(registry_path)
    corpus_file = Path(corpus_path) if corpus_path is not None else None
    categories = _registry_categories(_read_json(registry_file))
    files = {
        "input": _file_entry(input_file),
        "registry": _file_entry(registry_file),
    }
    if corpus_file is not None:
        files["corpus"] = _file_entry(corpus_file)
    for name, path in (behavior_files or {}).items():
        files[str(name)] = _file_entry(Path(path))
    default_loop_config = {
        "max_turns": 8,
        "max_identical_calls": 2,
        "action_retries": 1,
        "forced_finalize_retries": 1,
        "repetition_retries": 1,
    }
    default_loop_config.update(dict(loop_config or {}))
    return {
        "dataset_id": dataset_id,
        "sample_count": sample_count,
        "seed": seed,
        "input_fields": ["domain", "field_name"],
        "category_count": len(categories),
        "approved_levels": ["L1", "L2", "L3", "L4"],
        "code_revision": code_revision,
        "files": files,
        "model": dict(model or {"loaded": False, "backend": "scripted-cpu"}),
        "loop_config": default_loop_config,
        "working_tree_dirty": working_tree_dirty,
    }


def _record_from_row(row: Mapping[str, Any]) -> MetadataRecord:
    raw = row.get("record")
    source = dict(raw) if isinstance(raw, Mapping) else dict(row)
    return MetadataRecord.from_mapping(
        {
            "domain": source.get("domain"),
            "field_name": source.get("field_name")
            or (source.get("metadata") or {}).get("field_name"),
            "source_id": row.get("source_id"),
        }
    )


def _trace_tool_stats(trace: LoopTrace) -> tuple[int, int]:
    observations = [event for event in trace.events if event.kind == "observation"]
    successes = sum(event.data.get("status") == "success" for event in observations)
    return len(observations), successes


def _macro_f1(scored: Sequence[Mapping[str, Any]]) -> float:
    labels = sorted(
        {
            str(value)
            for row in scored
            for value in (row.get("ground_truth"), row.get("predicted_category"))
            if value is not None
        }
    )
    if not labels:
        return 0.0
    scores: list[float] = []
    for label in labels:
        true_positive = sum(
            row.get("ground_truth") == label and row.get("predicted_category") == label
            for row in scored
        )
        false_positive = sum(
            row.get("ground_truth") != label and row.get("predicted_category") == label
            for row in scored
        )
        false_negative = sum(
            row.get("ground_truth") == label and row.get("predicted_category") != label
            for row in scored
        )
        denominator = (2 * true_positive) + false_positive + false_negative
        scores.append((2 * true_positive) / denominator if denominator else 0.0)
    return sum(scores) / len(scores)


def evaluate_agent_rows(
    rows: Iterable[Mapping[str, Any]],
    controller_factory: ControllerFactory,
) -> dict[str, Any]:
    """Run loops while keeping evaluation labels outside controller state."""

    scored: list[dict[str, Any]] = []
    total_calls = 0
    successful_calls = 0
    for index, raw in enumerate(rows):
        row = dict(raw)
        record = _record_from_row(row)
        trace = controller_factory(row).run(record)
        calls, successes = _trace_tool_stats(trace)
        total_calls += calls
        successful_calls += successes
        final = trace.final_result or {}
        predicted_category = final.get("category")
        predicted_level = final.get("grading_level")
        expected_category = row.get("ground_truth")
        expected_level = row.get("ground_truth_level")
        category_correct = bool(trace.terminated and predicted_category == expected_category)
        level_correct = bool(trace.terminated and predicted_level == expected_level)
        scored.append(
            {
                "index": index,
                "source_id": row.get("source_id"),
                "ground_truth": expected_category,
                "ground_truth_level": expected_level,
                "predicted_category": predicted_category,
                "predicted_level": predicted_level,
                "category_correct": category_correct,
                "level_correct": level_correct,
                "joint_correct": category_correct and level_correct,
                "stop_reason": trace.stop_reason,
                "trace": trace.to_dict(),
            }
        )

    total = len(scored)
    terminal = sum(bool(row["trace"]["terminated"]) for row in scored)
    category_correct = sum(bool(row["category_correct"]) for row in scored)
    level_correct = sum(bool(row["level_correct"]) for row in scored)
    joint_correct = sum(bool(row["joint_correct"]) for row in scored)
    retry_count = sum(
        event["kind"] == "retry"
        for row in scored
        for event in row["trace"]["events"]
    )
    repetition_interventions = sum(
        event["kind"] == "controller_intervention"
        and event["data"].get("reason") == "repetition_limit"
        for row in scored
        for event in row["trace"]["events"]
    )
    format_failures = sum(
        event["kind"] in {"retry", "rejected_action"}
        and event["data"].get("reason") == "invalid_action"
        for row in scored
        for event in row["trace"]["events"]
    )
    legal_action_failures = sum(
        event["kind"] in {"retry", "rejected_action"}
        and event["data"].get("reason") in {"uncallable_skill", "invalid_arguments"}
        for row in scored
        for event in row["trace"]["events"]
    )
    early_stop_attempts = sum(
        event["kind"] in {"retry", "rejected_action"}
        and event["data"].get("reason") == "stop_without_final_result"
        for row in scored
        for event in row["trace"]["events"]
    )
    accepted_action_turns = sum(
        event["kind"] == "policy_action"
        or (
            event["kind"] == "controller_intervention"
            and event["data"].get("reason") == "repetition_limit"
        )
        for row in scored
        for event in row["trace"]["events"]
    )
    rejected_action_turns = sum(
        event["kind"] == "rejected_action"
        for row in scored
        for event in row["trace"]["events"]
    )
    action_turns = accepted_action_turns + rejected_action_turns
    policy_actions = sum(
        event["kind"] == "policy_action"
        for row in scored
        for event in row["trace"]["events"]
    )
    parsed_but_rejected = legal_action_failures + repetition_interventions
    policy_response_attempts = (
        policy_actions
        + parsed_but_rejected
        + format_failures
        + early_stop_attempts
    )
    parsed_policy_responses = policy_actions + parsed_but_rejected
    terminal_validation_failures = sum(
        event["kind"] == "retry" and event["data"].get("reason") == "invalid_terminal"
        for row in scored
        for event in row["trace"]["events"]
    )
    forced_finalizations = sum(
        event["kind"] == "policy_action" and bool(event["data"].get("forced_finalize"))
        for row in scored
        for event in row["trace"]["events"]
    )
    forced_finalization_missing_evidence = sum(
        event["kind"] == "controller_intervention"
        and event["data"].get("reason") == "forced_finalization_missing_evidence"
        for row in scored
        for event in row["trace"]["events"]
    )
    forced_terminations = 0
    normal_terminations = 0
    retrieval_hits = 0
    retrieval_rows = 0
    neighbor_field_call_rows = 0
    neighbor_field_calls = 0
    description_skill_call_rows = 0
    description_skill_calls = 0
    required_input_violations = 0
    total_tokens = 0
    measured_elapsed_seconds = 0.0
    measured_model_calls = 0
    skill_correction_attempts = 0
    for row in scored:
        events = row["trace"]["events"]
        row_neighbor_calls = sum(
            event["kind"] == "observation"
            and event["data"].get("skill") == "GET_TABLE_FIELDS"
            for event in events
        )
        neighbor_field_calls += row_neighbor_calls
        neighbor_field_call_rows += row_neighbor_calls > 0
        row_description_calls = sum(
            event["kind"] == "observation"
            and event["data"].get("skill") in {
                "INTERPRET_FIELD_DESCRIPTION", "INTERPRET_TABLE_DESCRIPTION"
            }
            for event in events
        )
        description_skill_calls += row_description_calls
        description_skill_call_rows += row_description_calls > 0
        if row["trace"]["terminated"]:
            final_actions = [
                event
                for event in events
                if event["kind"] == "policy_action"
                and event["data"].get("call", {}).get("tool") == "FINALIZE_CLASSIFICATION"
            ]
            if final_actions and final_actions[-1]["data"].get("forced_finalize"):
                forced_terminations += 1
            else:
                normal_terminations += 1
        retrieval = [
            event for event in events
            if event["kind"] == "observation"
            and event["data"].get("skill") in {"SEARCH_STANDARD", "SELECT_CATEGORY_TOPK"}
            and event["data"].get("status") == "success"
        ]
        if retrieval:
            retrieval_rows += 1
            candidates = retrieval[-1]["data"].get("data", {}).get("standard_candidates", [])
            expected = row.get("ground_truth")
            if any(
                isinstance(candidate, Mapping)
                and (candidate.get("category") or candidate.get("category_id")) == expected
                for candidate in candidates
            ):
                retrieval_hits += 1
        for event in events:
            metadata = event["data"].get("metadata", {})
            if event["kind"] == "observation" and metadata.get("required_input_violation"):
                required_input_violations += 1
            if event["kind"] == "observation":
                skill_correction_attempts += max(
                    int(metadata.get("format_retry_count", int(metadata.get("model_call_count") or 1) - 1)), 0
                )
            token_usage = metadata.get("token_usage")
            if isinstance(token_usage, Mapping):
                total_tokens += int(token_usage.get("total_tokens") or 0)
            elapsed = metadata.get("elapsed_seconds")
            if isinstance(elapsed, (int, float)):
                measured_elapsed_seconds += float(elapsed)
                measured_model_calls += int(metadata.get("model_call_count") or 1)
    return {
        "total": total,
        "terminal_results": terminal,
        "termination_rate": terminal / total if total else 0.0,
        "tool_calls": total_calls,
        "successful_tool_calls": successful_calls,
        "tool_execution_success_rate": successful_calls / total_calls if total_calls else 1.0,
        "category_correct": category_correct,
        "level_correct": level_correct,
        "joint_correct": joint_correct,
        "category_accuracy": category_correct / total if total else 0.0,
        "level_accuracy": level_correct / total if total else 0.0,
        "joint_accuracy": joint_correct / total if total else 0.0,
        "category_macro_f1": _macro_f1(scored),
        "average_tool_calls": total_calls / total if total else 0.0,
        "neighbor_field_calls": neighbor_field_calls,
        "neighbor_field_call_rows": neighbor_field_call_rows,
        "neighbor_field_call_rate": neighbor_field_call_rows / total if total else 0.0,
        "description_skill_calls": description_skill_calls,
        "description_skill_call_rows": description_skill_call_rows,
        "description_skill_call_rate": description_skill_call_rows / total if total else 0.0,
        "action_turns": action_turns,
        "accepted_action_turns": accepted_action_turns,
        "action_parse_rate_after_retry": (
            accepted_action_turns / action_turns if action_turns else 0.0
        ),
        "policy_response_attempts": policy_response_attempts,
        "json_object_parse_rate": (
            parsed_policy_responses / policy_response_attempts
            if policy_response_attempts
            else 0.0
        ),
        "legal_action_rate_given_parsed": (
            policy_actions / parsed_policy_responses if parsed_policy_responses else 0.0
        ),
        "retry_count": retry_count,
        "format_failures": format_failures,
        "legal_action_failures": legal_action_failures,
        "early_stop_attempts": early_stop_attempts,
        "terminal_validation_failures": terminal_validation_failures,
        "early_stop_recoveries": sum(
            bool(row["trace"]["terminated"])
            and any(
                event["data"].get("reason") == "stop_without_final_result"
                for event in row["trace"]["events"]
                if event["kind"] == "retry"
            )
            for row in scored
        ),
        "repetition_interventions": repetition_interventions,
        "repetition_recoveries": sum(
            bool(row["trace"]["terminated"])
            and any(
                event["kind"] == "controller_intervention"
                and event["data"].get("reason") == "repetition_limit"
                and bool(event["data"].get("recovered"))
                for event in row["trace"]["events"]
            )
            for row in scored
        ),
        "forced_finalizations": forced_finalizations,
        "forced_finalization_missing_evidence": forced_finalization_missing_evidence,
        "normal_terminations": normal_terminations,
        "forced_terminations": forced_terminations,
        "retrieval_recall_at_5": retrieval_hits / retrieval_rows if retrieval_rows else 0.0,
        "retrieval_evaluated_rows": retrieval_rows,
        "required_input_violations_reaching_execution": required_input_violations,
        "total_tokens": total_tokens if measured_model_calls else None,
        "measured_model_calls": measured_model_calls,
        "skill_correction_attempts": skill_correction_attempts,
        "measured_elapsed_seconds": (
            measured_elapsed_seconds if measured_model_calls else None
        ),
        "stop_reasons": {
            reason: sum(row["stop_reason"] == reason for row in scored)
            for reason in sorted({str(row["stop_reason"]) for row in scored})
        },
        "rows": scored,
    }


def write_run_artifacts(
    output_dir: str | Path,
    manifest: Mapping[str, Any],
    report: Mapping[str, Any],
    *,
    classification_mode: str = "legacy",
) -> ArtifactPaths:
    """Write manifest, one trace row per sample, and aggregate summary."""

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    paths = ArtifactPaths(
        manifest=root / "manifest.json",
        traces=root / "traces.jsonl",
        summary=root / "summary.json",
        observations=root / "observations.jsonl" if classification_mode == "two-stage" else None,
        model_calls=root / "model_calls.jsonl" if classification_mode == "two-stage" else None,
        trajectory=root / "trajectory.jsonl" if classification_mode == "two-stage" else None,
        evaluation=root / "evaluation.jsonl" if classification_mode == "two-stage" else None,
    )
    rows = report.get("rows", [])
    projected = [project_trace(row) for row in rows] if classification_mode == "two-stage" else []
    _atomic_write_text(
        paths.manifest, json.dumps(dict(manifest), ensure_ascii=False, indent=2) + "\n"
    )
    _atomic_write_text(paths.traces, _jsonl_text(rows))
    if classification_mode == "two-stage":
        for name, target in (
            ("observations", paths.observations),
            ("model_calls", paths.model_calls),
            ("trajectory", paths.trajectory),
            ("evaluation", paths.evaluation),
        ):
            assert target is not None
            records = []
            for item in projected:
                payload = item[name]
                records.extend(payload if isinstance(payload, list) else [payload])
            _atomic_write_text(target, _jsonl_text(records))
    summary = {key: value for key, value in report.items() if key != "rows"}
    _atomic_write_text(paths.summary, json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return paths


def _jsonl_text(rows: Iterable[Mapping[str, Any]]) -> str:
    return "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)


def _atomic_write_text(path: Path, content: str) -> None:
    """Replace one completed artifact atomically within its output directory."""

    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="\n", dir=path.parent,
        prefix=f".{path.name}.", suffix=".tmp", delete=False,
    ) as handle:
        temp_path = Path(handle.name)
        try:
            handle.write(content)
        except BaseException:
            temp_path.unlink(missing_ok=True)
            raise
    try:
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def _git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _git_dirty() -> bool | None:
    try:
        output = subprocess.check_output(
            ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
        )
        return bool(output.strip())
    except (OSError, subprocess.CalledProcessError):
        return None


def _skill_behavior_files(
    skill_root: Path, *, classification_mode: str = "legacy",
    include_metadata_extra: bool = False,
    include_table_fields: bool = False,
    include_description_skills: bool = False,
) -> dict[str, Path]:
    files = {
        f"skill:{path.name}": path
        for path in sorted(skill_root.glob("*.md"))
        if path.is_file()
    }
    if classification_mode == "two-stage":
        files.update({
            f"skill:two_stage/{path.name}": path
            for path in sorted((skill_root / "two_stage").glob("*.md"))
            if path.is_file()
        })
    if include_metadata_extra:
        files.update({
            f"skill:metadata_extra/{path.name}": path
            for path in (skill_root / "metadata_extra" / "metadata_acquisition_GET_FIELD_METADATA.md",)
            if path.is_file()
        })
    if include_table_fields:
        path = skill_root / "neighbor_extra" / "metadata_acquisition_GET_TABLE_FIELDS.md"
        if path.is_file():
            files[f"skill:neighbor_extra/{path.name}"] = path
    if include_description_skills:
        for path in sorted((skill_root / "semantic_extra").glob("*.md")):
            files[f"skill:semantic_extra/{path.name}"] = path
    return files


def _observation(raw: Mapping[str, Any], tool: str) -> Observation:
    return Observation(
        skill=str(raw.get("skill") or tool),
        status=str(raw.get("status") or "success"),
        summary=str(raw.get("summary") or tool),
        data=dict(raw.get("data") or {}),
    )


def _scripted_factory(
    scripts: Mapping[str, Any],
    skill_root: Path,
    registry_categories: Sequence[str],
) -> ControllerFactory:
    catalog = dynamic_loop_catalog()

    def factory(row: Mapping[str, Any]) -> AgentLoopController:
        source_id = str(row.get("source_id"))
        sample = scripts.get(source_id)
        if not isinstance(sample, Mapping):
            raise ValueError(f"missing scripted responses for source_id={source_id!r}")
        raw_outputs = sample.get("observations", {})
        if not isinstance(raw_outputs, Mapping):
            raise ValueError(f"observations for {source_id!r} must be an object")
        outputs = {
            str(tool): [_observation(item, str(tool)) for item in values]
            for tool, values in raw_outputs.items()
        }
        config_raw = sample.get("config", {})
        return AgentLoopController(
            policy=ScriptedLoopPolicy(sample.get("actions", [])),
            executor=ScriptedSkillExecutor(outputs),
            catalog=catalog,
            skill_loader=SkillLoader(skill_root, catalog),
            config=LoopConfig(**dict(config_raw)),
            registry_categories=registry_categories,
        )

    return factory


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--backend", choices=("scripted", "transformers"), default="scripted")
    parser.add_argument("--responses", type=Path)
    parser.add_argument("--policy", choices=("fixed", "dynamic"), default="dynamic")
    parser.add_argument("--classification-mode", choices=("legacy", "two-stage"), default="legacy")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--few-shot", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--allow-remote-model-files", action="store_true")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--max-identical-calls", type=int, default=2)
    parser.add_argument("--action-retries", type=int, default=1)
    parser.add_argument("--forced-finalize-retries", type=int, default=1)
    parser.add_argument("--repetition-retries", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--metadata-source", type=Path)
    parser.add_argument("--neighbor-fields", action="store_true")
    parser.add_argument("--neighbor-prefetch", action="store_true")
    parser.add_argument("--description-skills", action="store_true")
    parser.add_argument("--topk-evidence-mode", choices=("augmented", "raw", "raw-flat"), default="augmented")
    parser.add_argument("--topk-selection-mode", choices=("single", "balanced-vote"), default="single")
    parser.add_argument("--topk-selection-seed", type=int, default=20260930)
    parser.add_argument("--skill-root", type=Path, default=Path("skills"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--pilot-count", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.topk_selection_mode == "balanced-vote" and (
        args.classification_mode != "two-stage" or args.policy != "dynamic" or args.backend != "transformers"
    ):
        raise ValueError("balanced-vote requires transformers and dynamic two-stage classification")
    if args.topk_evidence_mode in {"raw", "raw-flat"} and not args.description_skills:
        raise ValueError("raw Top-K evidence mode requires --description-skills")
    if args.neighbor_prefetch and not args.neighbor_fields:
        raise ValueError("--neighbor-prefetch requires --neighbor-fields")
    if args.neighbor_fields and args.metadata_source is None:
        raise ValueError("--neighbor-fields requires --metadata-source")
    if args.neighbor_fields and args.backend != "transformers":
        raise ValueError("--neighbor-fields requires the transformers backend")
    if args.neighbor_fields and args.policy != "dynamic":
        raise ValueError("--neighbor-fields currently requires --policy dynamic")
    if args.description_skills and (
        args.metadata_source is None or args.backend != "transformers"
        or args.policy != "dynamic" or args.classification_mode != "two-stage"
    ):
        raise ValueError("--description-skills requires transformers, dynamic two-stage, and --metadata-source")
    if args.classification_mode == "two-stage" and args.backend != "transformers":
        raise ValueError("two-stage requires the transformers backend")
    if args.classification_mode == "two-stage" and args.policy != "dynamic":
        raise ValueError("two-stage currently supports only the dynamic policy")
    prior_manifest = args.output / "manifest.json"
    if args.resume and not prior_manifest.is_file() and args.output.is_dir():
        prior_results = (
            "traces.jsonl", "summary.json", "observations.jsonl", "model_calls.jsonl",
            "trajectory.jsonl", "evaluation.jsonl",
        )
        if any((args.output / name).is_file() and (args.output / name).stat().st_size > 0
               for name in prior_results):
            raise ValueError("cannot resume existing results without manifest.json")
    if args.resume and prior_manifest.is_file():
        previous = _read_json(prior_manifest)
        model = previous.get("model", {}) if isinstance(previous, Mapping) else {}
        prior_selection = model.get("topk_selection_mode", "single") if isinstance(model, Mapping) else "single"
        prior_seed = model.get("topk_selection_seed", 20260930) if isinstance(model, Mapping) else 20260930
        if prior_selection != args.topk_selection_mode:
            raise ValueError("cannot resume artifacts from a different Top-K selection mode")
        if prior_seed != args.topk_selection_seed:
            raise ValueError("cannot resume artifacts from a different Top-K selection seed")
        prior_mode = model.get("classification_mode", "legacy") if isinstance(model, Mapping) else "legacy"
        if prior_mode != args.classification_mode:
            raise ValueError("cannot resume artifacts from a different classification mode")
        prior_neighbors = bool(model.get("neighbor_fields_dsl_enabled", False)) if isinstance(model, Mapping) else False
        if prior_neighbors != args.neighbor_fields:
            raise ValueError("cannot resume artifacts from a different neighbor-fields mode")
        prior_prefetch = model.get("neighbor_fields_mode") == "prefetch" if isinstance(model, Mapping) else False
        if prior_prefetch != args.neighbor_prefetch:
            raise ValueError("cannot resume artifacts from a different neighbor prefetch mode")
        prior_descriptions = bool(model.get("description_skills_enabled", False)) if isinstance(model, Mapping) else False
        if prior_descriptions != args.description_skills:
            raise ValueError("cannot resume artifacts from a different description-Skill mode")
        prior_topk_mode = model.get("topk_evidence_mode", "augmented") if isinstance(model, Mapping) else "augmented"
        if prior_topk_mode != args.topk_evidence_mode:
            raise ValueError("cannot resume artifacts from a different Top-K evidence mode")
        files = previous.get("files", {}) if isinstance(previous, Mapping) else {}
        prior_has_source = isinstance(files, Mapping) and "metadata_source" in files
        current_has_source = args.metadata_source is not None
        if prior_has_source != current_has_source:
            raise ValueError("cannot resume artifacts from a different metadata source configuration")
        if current_has_source:
            source_entry = files.get("metadata_source", {}) if isinstance(files, Mapping) else {}
            prior_hash = source_entry.get("sha256") if isinstance(source_entry, Mapping) else None
            if not isinstance(prior_hash, str) or prior_hash != _sha256(args.metadata_source):
                source_kind = "neighbor metadata" if args.neighbor_fields else "metadata"
                raise ValueError(f"cannot resume artifacts from a different {source_kind} source")
    rows = _read_rows(args.input)
    if args.pilot_count is not None:
        rows = select_pilot_rows(rows, count=args.pilot_count, seed=args.seed)
    intended_sample_count = len(rows)
    traces_path = args.output / "traces.jsonl"
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        raise FileExistsError(f"output directory is not empty: {args.output}; use --resume")
    existing_scored = _read_rows(traces_path) if args.resume and traces_path.is_file() else []
    if args.resume:
        rows = pending_rows_for_resume(rows, traces_path)
    registry_value = _read_json(args.registry)
    registry_categories = _registry_categories(registry_value)
    registry_entries = _registry_entry_map(registry_value)
    loop_config = LoopConfig(
        max_turns=args.max_turns,
        max_identical_calls=args.max_identical_calls,
        action_retries=args.action_retries,
        forced_finalize_retries=args.forced_finalize_retries,
        repetition_retries=args.repetition_retries,
    )
    behavior_files = _skill_behavior_files(
        args.skill_root, classification_mode=args.classification_mode,
        include_metadata_extra=args.backend == "transformers" and args.metadata_source is not None,
        include_table_fields=args.neighbor_fields,
        include_description_skills=args.description_skills,
    )
    if args.backend == "scripted":
        if args.responses is None:
            raise ValueError("--responses is required for the scripted backend")
        response_payload = _read_json(args.responses)
        if not isinstance(response_payload, Mapping) or not isinstance(
            response_payload.get("samples"), Mapping
        ):
            raise ValueError("responses must contain a samples object")
        behavior_files["scripted_responses"] = args.responses
        factory = _scripted_factory(
            response_payload["samples"],
            args.skill_root,
            registry_categories,
        )
        model_manifest: dict[str, Any] = {
            "loaded": False,
            "backend": "scripted-cpu",
            "sample_config_overrides": {
                str(source_id): dict(sample.get("config") or {})
                for source_id, sample in response_payload["samples"].items()
                if isinstance(sample, Mapping) and sample.get("config")
            },
        }
    else:
        if args.model is None:
            raise ValueError("--model is required for the transformers backend")
        from agent.orchestration.planner import TransformersTextGenerator

        generate = TransformersTextGenerator(
            args.model,
            max_new_tokens=args.max_new_tokens,
            local_files_only=not args.allow_remote_model_files,
            enable_thinking=args.enable_thinking,
        )
        few_shot_messages = load_few_shot_messages(args.few_shot) if args.few_shot else ()
        metadata_lookup = (
            FieldMetadataLookup.from_canonical(args.metadata_source)
            if args.metadata_source else None
        )
        table_field_lookup = (
            TableFieldLookup.from_canonical(args.metadata_source)
            if args.neighbor_fields else None
        )
        if args.metadata_source:
            behavior_files["metadata_source"] = args.metadata_source
        if args.few_shot:
            behavior_files["few_shot"] = args.few_shot

        def factory(_row: Mapping[str, Any]) -> AgentLoopController:
            return build_model_controller(
                generate=generate,
                policy_mode=args.policy,
                skill_root=args.skill_root,
                registry_entries=registry_entries,
                config=loop_config,
                few_shot_messages=few_shot_messages,
                top_k=args.top_k,
                metadata_lookup=metadata_lookup,
                table_field_lookup=table_field_lookup,
                neighbor_prefetch=args.neighbor_prefetch,
                include_description_skills=args.description_skills,
                topk_evidence_mode=args.topk_evidence_mode,
                topk_selection_mode=args.topk_selection_mode,
                topk_selection_seed=args.topk_selection_seed,
                classification_mode=args.classification_mode,
            )

        model_manifest = {
            "loaded": True,
            "backend": "transformers",
            "name_or_path": str(args.model),
            "local_files_only": not args.allow_remote_model_files,
            "enable_thinking": args.enable_thinking,
            "max_new_tokens": args.max_new_tokens,
            "do_sample": False,
            "policy": args.policy,
            "few_shot": bool(args.few_shot),
            "retrieval": {
                "backend": (
                    "model-full-catalog-short-options" if args.classification_mode == "two-stage"
                    else "deterministic-lexical"
                ),
                "top_k": args.top_k,
                "selection_id_scheme": (
                    "sorted-canonical-c001-v1"
                    if args.classification_mode == "two-stage" else None
                ),
            },
            "classification_mode": args.classification_mode,
            "metadata_dsl_enabled": metadata_lookup is not None,
            "neighbor_fields_dsl_enabled": table_field_lookup is not None,
            "description_skills_enabled": args.description_skills,
            "topk_evidence_mode": args.topk_evidence_mode,
            "neighbor_fields_mode": (
                "prefetch" if args.neighbor_prefetch else
                "optional" if table_field_lookup else "off"
            ),
            "neighbor_fields_limit": table_field_lookup.max_fields if table_field_lookup else None,
            "neighbor_fields_source_policy": (
                "cross-split-unlabeled-table-metadata" if table_field_lookup else None
            ),
        }
    from agent.orchestration.topk_selection import SELECTION_POLICY

    model_manifest.update({
        "topk_selection_mode": args.topk_selection_mode,
        "topk_selection_seed": args.topk_selection_seed,
        "topk_selection_algorithm": SELECTION_POLICY if args.topk_selection_mode == "balanced-vote" else "single-with-one-format-retry",
        "topk_selection_rounds": 3 if args.topk_selection_mode == "balanced-vote" else 1,
        "topk_selection_quorum": 2 if args.topk_selection_mode == "balanced-vote" else 1,
    })
    report = evaluate_agent_rows(rows, factory)
    if existing_scored:
        report = _recalculate_saved_rows([*existing_scored, *report["rows"]])
    manifest = build_run_manifest(
        dataset_id=args.dataset_id,
        input_path=args.input,
        registry_path=args.registry,
        corpus_path=args.corpus,
        code_revision=_git_revision(),
        sample_count=intended_sample_count,
        seed=args.seed,
        behavior_files=behavior_files,
        model=model_manifest,
        loop_config=loop_config.__dict__,
        working_tree_dirty=_git_dirty(),
    )
    paths = write_run_artifacts(args.output, manifest, report,
                                classification_mode=args.classification_mode)
    print(
        json.dumps(
            {
                "manifest": str(paths.manifest),
                "traces": str(paths.traces),
                "summary": str(paths.summary),
                **{key: value for key, value in report.items() if key != "rows"},
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ArtifactPaths",
    "build_model_controller",
    "build_run_manifest",
    "evaluate_agent_rows",
    "load_experiment_variants",
    "main",
    "pending_rows_for_resume",
    "select_pilot_rows",
    "write_run_artifacts",
]
