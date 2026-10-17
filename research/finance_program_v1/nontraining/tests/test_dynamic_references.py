"""Dynamic values must come from current state; only model I/O is scripted."""
import json

import pytest

from test_program_loop import FULL, REGISTRY, ROWS, Script, api, program, run
from program_loop.validation import canonical_reference, parse_program


@pytest.fixture
def controller():
    return api().build_program_controller(Script([]), REGISTRY, ROWS)


@pytest.mark.parametrize("value", [{}, [], None, 7, True, "handwritten", "$state.wrong"])
def test_every_dynamic_catalog_input_rejects_noncanonical_explicit_values(controller, value):
    catalog = controller.bridge.catalog
    available = {key for spec in catalog.values() for key in (*spec.required_inputs, *spec.optional_inputs, *spec.produces)}
    for spec in catalog.values():
        for key in (*spec.required_inputs, *spec.optional_inputs):
            expected = canonical_reference(key, available)
            if not expected.startswith("$state."):
                continue
            raw = program({"tool": spec.name, "arguments": {key: value}})
            with pytest.raises(ValueError, match=f"step 0: incorrect canonical reference for {key}; expected") as error:
                parse_program(raw, catalog, available, 12)
            assert expected in str(error.value)


def test_catalog_policy_and_prompt_explain_dynamic_values(controller):
    from program_loop.planner import SYSTEM
    for item in controller.library:
        for details in item["planning"]["inputs"].values():
            expected = "canonical_reference_only" if details["canonical_reference"].startswith("$state.") else "initial_binding_contract"
            assert details["explicit_value_policy"] == expected
    assert "explicit_value_policy" in SYSTEM
    assert "Never handwrite dynamic state values" in SYSTEM


def test_parser_preserves_omissions_and_canonical_aliases(controller):
    raw = program("INTERPRET_FIELD_NAME", {"tool": "SELECT_CATEGORY_TOPK", "arguments": {"semantic_query": "$state.semantic_query"}},
                  {"tool": "INFER_CATEGORY_FROM_TOPK", "arguments": {"candidate_standard_categories": "$state.standard_candidates"}},
                  "INFER_GRADING_LEVEL", {"tool": "FINALIZE_CLASSIFICATION", "arguments": {"final_evidence_state": "$state.final_evidence_state"}})
    parsed = parse_program(raw, controller.bridge.catalog, {}, 12)
    assert parsed.to_dict() == json.loads(raw)
    assert parsed.steps[0].arguments == {}
    assert "field_abbreviation_semantics" not in parsed.steps[0].arguments


def test_initial_literals_remain_legal_in_parser(controller):
    raw = program({"tool": "INTERPRET_FIELD_NAME", "arguments": {"field_name": "balance"}},
                  {"tool": "INTERPRET_DOMAIN", "arguments": {"domain": "synthetic domain"}},
                  {"tool": "GET_FIELD_METADATA", "arguments": {"current_record": {"field_name": "balance"}}})
    assert parse_program(raw, controller.bridge.catalog, {}, 12).to_dict() == json.loads(raw)


def test_unavailable_optional_remains_rejected(controller):
    raw = program({"tool": "INTERPRET_FIELD_NAME", "arguments": {"field_abbreviation_semantics": "$state.field_abbreviation_semantics"}})
    with pytest.raises(ValueError, match="step 0: missing dependency field_abbreviation_semantics"):
        parse_program(raw, controller.bridge.catalog, {}, 12)


def test_explicit_available_optional_uses_canonical_path(controller):
    raw = program({"tool": "INTERPRET_FIELD_NAME", "arguments": {"field_abbreviation_semantics": "$state.field_abbreviation_semantics"}})
    assert parse_program(raw, controller.bridge.catalog, {"field_abbreviation_semantics": "observed"}, 12).to_dict() == json.loads(raw)


@pytest.mark.parametrize("key,value", [("semantic_query", {}), ("semantic_query", "observed text"), ("semantic_query", None)])
def test_controller_rejects_dynamic_literal_before_any_execution_or_program_budget(key, value):
    raw = program("INTERPRET_FIELD_NAME", {"tool": "SELECT_CATEGORY_TOPK", "arguments": {key: value}})
    _, trace, script = run([raw], max_planner_attempts=1)
    assert trace.planner_attempts[0]["status"] == "rejected"
    assert trace.planner_attempts[0]["raw_response"] == raw
    assert f"step 1: incorrect canonical reference for {key}; expected $state.{key}" in trace.planner_attempts[0]["error"]
    assert trace.budget == {"programs": 0, "planner_attempts": 1, "skill_steps": 0}
    assert trace.programs == [] and script.kinds == ["planner"]


def test_canonical_future_query_binds_after_semantic_update():
    raw = program("INTERPRET_DOMAIN", "INTERPRET_FIELD_NAME", {"tool": "SELECT_CATEGORY_TOPK", "arguments": {"semantic_query": "$state.semantic_query"}}, *FULL[2:])
    _, trace, script = run([raw])
    assert trace.terminated
    assert trace.programs[0]["steps"][2]["status"] == "success"
    query_messages = next(messages for messages in script.messages if "selected Skill SELECT_CATEGORY_TOPK." in messages[0]["content"])
    assert "domain_semantics" in json.dumps(query_messages)
    assert "field_semantics" in json.dumps(query_messages)
