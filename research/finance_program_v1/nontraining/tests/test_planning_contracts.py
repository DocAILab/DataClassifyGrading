"""Planning documentation plus real parsing/binding/guards; only model I/O is scripted."""
import importlib
import json

import pytest

from test_program_loop import FULL, REGISTRY, ROWS, Script, api, program, run


@pytest.fixture
def controller():
    module = api()
    return module.build_program_controller(Script([]), REGISTRY, ROWS)


def test_library_documents_canonical_paths_and_distinct_dependencies(controller):
    by_name = {item["name"]: item for item in controller.library}
    assert len(by_name) == 14
    name_inputs = by_name["INTERPRET_FIELD_NAME"]["planning"]["inputs"]
    assert by_name["INTERPRET_FIELD_NAME"]["planning"]["argument_template"] == {}
    assert name_inputs["field_name"]["canonical_reference"] == "$metadata.field_name"
    assert name_inputs["field_name"]["source"] == "initial_binding"
    assert name_inputs["field_name"]["possible_producers"] == []
    assert name_inputs["field_abbreviation_semantics"]["required"] is False
    assert name_inputs["field_abbreviation_semantics"]["possible_producers"] == ["EXPAND_FIELD_ABBREVIATION"]
    disambiguate_inputs = by_name["DISAMBIGUATE_FIELD_SEMANTICS"]["planning"]["inputs"]
    assert disambiguate_inputs["field_semantics"]["required"] is True
    assert disambiguate_inputs["field_semantics"]["possible_producers"] == ["INTERPRET_FIELD_NAME"]
    top5_inputs = by_name["SELECT_CATEGORY_TOPK"]["planning"]["inputs"]
    assert top5_inputs["semantic_query"]["canonical_reference"] == "$state.semantic_query"
    compare_inputs = by_name["COMPARE_STANDARD_CANDIDATES"]["planning"]["inputs"]
    assert compare_inputs["semantic_evidence"]["canonical_reference"] == "$state.semantic_evidence"
    assert "GET_FIELD_METADATA" in top5_inputs["semantic_query"]["possible_producers"]
    assert "GET_FIELD_METADATA" not in compare_inputs["semantic_evidence"]["possible_producers"]
    infer_inputs = by_name["INFER_CATEGORY_FROM_TOPK"]["planning"]["inputs"]
    assert infer_inputs["candidate_standard_categories"]["canonical_reference"] == "$state.standard_candidates"
    assert infer_inputs["candidate_standard_categories"]["possible_producers"] == ["SELECT_CATEGORY_TOPK"]
    final_inputs = by_name["FINALIZE_CLASSIFICATION"]["planning"]["inputs"]
    validation = importlib.import_module("program_loop.validation")
    assert final_inputs["final_evidence_state"]["requires_all"] == sorted(validation.FINAL_EVIDENCE_REQUIREMENTS)
    assert final_inputs["final_evidence_state"]["possible_producers"] == []
    json.dumps(controller.library, allow_nan=False)


def test_planning_interface_is_shared_and_preserves_original_library_contracts(controller):
    planning = importlib.import_module("program_loop.planning_contracts")
    validation = importlib.import_module("program_loop.validation")
    from pool_catalog import purpose
    from program_loop.catalog_context import skill_library

    catalog = controller.bridge.catalog
    for item in controller.library:
        spec = catalog[item["name"]]
        assert item["planning"] == planning.planning_contract(spec, catalog)
        assert {key: item[key] for key in item if key != "planning"} == {
            "name": spec.name, "purpose": purpose(spec.name), "role": spec.role,
            "required_inputs": list(spec.required_inputs), "optional_inputs": list(spec.optional_inputs),
            "produces": list(spec.produces), "terminal": spec.terminal,
            "contract": controller.bridge.loader.load(spec.name).body,
        }
        assert set(item["planning"]["inputs"]) == set((*spec.required_inputs, *spec.optional_inputs))
        assert item["planning"]["conditional_outputs"] == sorted(validation.possible_outputs(spec))
        for key, details in item["planning"]["inputs"].items():
            assert details["required"] == (key in spec.required_inputs)
            assert details["requires_all"] == (sorted(validation.FINAL_EVIDENCE_REQUIREMENTS) if key == "final_evidence_state" else [])
    assert skill_library(catalog, controller.bridge.loader) == controller.library


def test_shared_helpers_keep_aliases_conditional_outputs_and_invalidation(controller):
    validation = importlib.import_module("program_loop.validation")
    reference = validation.canonical_reference
    assert reference("current_record", set()) == "$record"
    assert reference("domain", set()) == "$domain"
    assert reference("target_field", set()) == "$metadata.field_name"
    assert reference("candidate_standard_categories", {"standard_candidates"}) == "$state.standard_candidates"
    assert reference("candidate_standard_categories", set()) is None
    assert reference("field_description", set()) is None
    assert reference("field_description", {"field_description"}) == "$state.field_description"
    catalog = controller.bridge.catalog
    metadata = validation.possible_outputs(catalog["GET_FIELD_METADATA"])
    assert {"field_metadata", "field_description", "current_metadata", "semantic_query"} <= metadata
    assert "semantic_evidence" not in metadata and "final_evidence_state" not in metadata
    assert {"semantic_query", "semantic_evidence"} <= validation.possible_outputs(catalog["INTERPRET_FIELD_NAME"])
    prerequisites = set(validation.FINAL_EVIDENCE_REQUIREMENTS)
    before = prerequisites | {"final_evidence_state", "final_result"}
    after = validation.advance_available(catalog["INTERPRET_DOMAIN"], before)
    assert before == prerequisites | {"final_evidence_state", "final_result"}  # Pure: no caller mutation.
    assert not ({"category", "grading_level", "aligned_standard_entry", "final_evidence_state", "final_result"} & after)
    assert "standard_candidates" in after
    after = validation.advance_available(catalog["SELECT_CATEGORY_TOPK"], before)
    assert "standard_candidates" in after and "category" not in after and "grading_level" not in after
    after = validation.advance_available(catalog["INFER_CATEGORY_FROM_TOPK"], before)
    assert {"category", "aligned_category", "candidate_category", "category_candidate"} <= after
    assert "grading_level" not in after and "final_evidence_state" not in after
    for missing in prerequisites:
        after = validation.advance_available(catalog["FINALIZE_CLASSIFICATION"], prerequisites - {missing})
        assert "final_evidence_state" not in after
    assert "final_evidence_state" in validation.advance_available(catalog["FINALIZE_CLASSIFICATION"], prerequisites)


def test_guidance_visible_in_one_plan_real_executor_success_without_inserted_skills():
    controller, trace, script = run([program(*FULL)])
    payload = json.loads(script.messages[0][1]["content"])
    assert all(item["planning"]["argument_template"] == {} for item in payload["skill_library"])
    assert type(controller.bridge.executor).__name__ == "PoolExecutor"
    assert type(controller.bridge.guard).__name__ == "PoolController"
    assert trace.terminated and trace.final_result == {"category": "cat1", "grading_level": "L2"}
    assert trace.budget == {"programs": 1, "planner_attempts": 1, "skill_steps": 5}
    assert [step["call"]["tool"] for step in trace.programs[0]["steps"]] == list(FULL)
    assert all(step["binding"]["version"] == "canonical-state-bindings-v1" for step in trace.programs[0]["steps"])
    assert payload["current"]["input"] == {"domain": "合成金融", "field_name": "balance"}
    messages = json.dumps(script.messages, ensure_ascii=False)
    for secret in ("PRIVATE-ROW-997", "SECRET-GOLD", "SECRET-GRADE", "PRELOADED-SECRET", "当前余额", "accounts"):
        assert secret not in messages


def test_description_semantics_does_not_supply_name_semantics(controller):
    from program_loop.validation import parse_program
    raw = program("INTERPRET_FIELD_DESCRIPTION", "DISAMBIGUATE_FIELD_SEMANTICS")
    with pytest.raises(ValueError, match="step 1: missing dependency field_semantics"):
        parse_program(raw, controller.bridge.catalog, {"field_description": "observed"}, 12)
    _, trace, _ = run([program("GET_FIELD_METADATA", "INTERPRET_FIELD_DESCRIPTION", "DISAMBIGUATE_FIELD_SEMANTICS")], max_planner_attempts=1)
    assert trace.planner_attempts[0]["status"] == "rejected"
    assert "missing dependency field_semantics" in trace.planner_attempts[0]["error"]
    assert trace.budget["skill_steps"] == 0


def test_absent_optional_null_is_rejected_and_raw_program_is_preserved():
    raw = program({"tool": "INTERPRET_FIELD_NAME", "arguments": {"field_abbreviation_semantics": None}})
    _, trace, _ = run([raw], max_planner_attempts=1)
    assert trace.planner_attempts[0]["status"] == "rejected"
    assert trace.planner_attempts[0]["error"] == "step 0: missing dependency field_abbreviation_semantics"
    assert trace.planner_attempts[0]["raw_response"] == raw
    assert trace.budget == {"programs": 0, "planner_attempts": 1, "skill_steps": 0}


@pytest.mark.parametrize("prefix,tool,key,wrong,expected", [
    ((), "INTERPRET_FIELD_NAME", "field_name", "$state.field_name", "$metadata.field_name"),
    (("INTERPRET_FIELD_NAME",), "SELECT_CATEGORY_TOPK", "semantic_query", "$state.semantic_evidence", "$state.semantic_query"),
    (("INTERPRET_FIELD_NAME", "SELECT_CATEGORY_TOPK"), "COMPARE_STANDARD_CANDIDATES", "semantic_evidence", "$state.semantic_query", "$state.semantic_evidence"),
])
def test_wrong_explicit_references_still_reject(prefix, tool, key, wrong, expected):
    raw = program(*prefix, {"tool": tool, "arguments": {key: wrong}})
    _, trace, _ = run([raw], max_planner_attempts=1)
    assert trace.planner_attempts[0]["status"] == "rejected"
    assert trace.planner_attempts[0]["error"] == f"step {len(prefix)}: incorrect canonical reference for {key}; expected {expected}"
    assert trace.planner_attempts[0]["raw_response"] == raw
    assert trace.budget["skill_steps"] == 0


def test_explicit_canonical_parameters_keep_real_binding():
    raw = program(
        {"tool": "INTERPRET_FIELD_NAME", "arguments": {"field_name": "$metadata.field_name"}},
        {"tool": "SELECT_CATEGORY_TOPK", "arguments": {"semantic_query": "$state.semantic_query"}},
        {"tool": "INFER_CATEGORY_FROM_TOPK", "arguments": {"candidate_standard_categories": "$state.standard_candidates"}},
        {"tool": "INFER_GRADING_LEVEL", "arguments": {"aligned_category": "$state.aligned_category"}},
        {"tool": "FINALIZE_CLASSIFICATION", "arguments": {"final_evidence_state": "$state.final_evidence_state"}},
    )
    _, trace, _ = run([raw])
    assert trace.terminated and trace.planner_attempts[0]["raw_response"] == raw
    assert all(step["status"] == "success" for step in trace.programs[0]["steps"])


def test_conditional_metadata_blocks_current_program_and_replans_independently():
    rows = [{"id": "PRIVATE-ROW-997", "domain": "合成金融", "metadata": {"field_name": "balance"}}]
    raw = program("GET_FIELD_METADATA", "INTERPRET_FIELD_DESCRIPTION", "INTERPRET_DOMAIN")
    _, trace, script = run([raw, program(*FULL)], rows=rows)
    first = trace.programs[0]
    assert first["status"] == "failed" and first["failure_index"] == 1
    assert [step["status"] for step in first["steps"]] == ["success", "blocked"]
    assert [step["call"]["tool"] for step in first["steps"]] == ["GET_FIELD_METADATA", "INTERPRET_FIELD_DESCRIPTION"]
    assert "field_description" not in trace.state
    assert trace.terminated and trace.budget == {"programs": 2, "planner_attempts": 2, "skill_steps": 7}
    assert [step["call"]["tool"] for step in trace.programs[1]["steps"]] == list(FULL)
    assert script.kinds[:3] == ["planner", "planner", "INTERPRET_FIELD_NAME"]
    feedback = json.loads(script.messages[1][1]["content"])["feedback"]
    assert feedback["status"] == "failed" and feedback["failure_index"] == 1
    assert trace.planner_attempts[0]["raw_response"] == raw


def test_prompt_documents_omission_dependencies_and_conditional_execution(controller):
    from program_loop.planner import SYSTEM
    assert "Prefer arguments:{}" in SYSTEM
    assert "binder supplies available required inputs" in SYSTEM
    assert "omitted optional inputs are not automatically inserted" in SYSTEM
    assert "omit" in SYSTEM and "null" in SYSTEM
    assert "possible_producers" in SYSTEM and "requires_all" in SYSTEM
    assert "semantic_query" in SYSTEM and "semantic_evidence" in SYSTEM
    assert "field_description_semantics" in SYSTEM and "field_semantics" in SYSTEM
    assert "conditional" in SYSTEM and "actual state" in SYSTEM
    assert "No Skill name is mandatory" in SYSTEM and "inserts no steps" in SYSTEM
    assert all(name not in SYSTEM for name in controller.bridge.catalog)


def test_real_binder_injects_required_but_not_omitted_available_optional(controller):
    from agent.orchestration.agent_loop import PolicyDecision
    from agent.orchestration.contracts import MetadataRecord, ToolCall

    record = MetadataRecord.from_mapping({"domain": "合成金融", "field_name": "balance"})
    state = {"field_abbreviation_semantics": "observed abbreviation interpretation"}
    controller.bridge.availability.initialize(record, state)
    spec = controller.bridge.catalog["INTERPRET_FIELD_NAME"]
    omitted = controller.bridge.guard._normalize_policy_decision(
        PolicyDecision(ToolCall(spec.name, {}), ""), record, state, spec)
    assert omitted.call.arguments == {"field_name": "$metadata.field_name"}
    explicit = controller.bridge.guard._normalize_policy_decision(
        PolicyDecision(ToolCall(spec.name, {"field_abbreviation_semantics": "$state.field_abbreviation_semantics"}), ""),
        record, state, spec)
    assert explicit.call.arguments == {
        "field_name": "$metadata.field_name",
        "field_abbreviation_semantics": "$state.field_abbreviation_semantics",
    }
