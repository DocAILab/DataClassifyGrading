import importlib
import json
import subprocess
import sys

import pytest


REGISTRY = {f"cat{i}": {"label": f"合成类别{i}", "description": f"无标签标准描述{i}", "grading_level": f"L{(i % 4) + 1}"} for i in range(1, 7)}
ROWS = [{"id": "PRIVATE-ROW-997", "domain": "合成金融", "metadata": {"field_name": "balance", "field_description": "当前余额", "table_name": "accounts"}, "gold_category": "SECRET-GOLD", "grading_level": "SECRET-GRADE"}]


def api():
    assert importlib.util.find_spec("program_loop") is not None, "Program orchestration package is not implemented"
    return importlib.import_module("program_loop")


def program(*tools):
    return json.dumps({"program": [{"tool": t, "arguments": {}} if isinstance(t, str) else t for t in tools]})


FULL = ("INTERPRET_FIELD_NAME", "SELECT_CATEGORY_TOPK", "INFER_CATEGORY_FROM_TOPK", "INFER_GRADING_LEVEL", "FINALIZE_CLASSIFICATION")


class Script:
    """Only model I/O is scripted; execution is the frozen production executor."""
    def __init__(self, plans, overrides=None):
        self.plans = iter(plans)
        self.messages = []
        self.kinds = []
        self.overrides = overrides or {}

    def __call__(self, messages):
        self.messages.append(messages)
        system = messages[0]["content"]
        if "Global Program Planner" in system:
            self.kinds.append("planner")
            response = next(self.plans)
        else:
            name = system.split("selected Skill ")[1].split(".")[0]
            self.kinds.append(name)
            if name in self.overrides:
                values = self.overrides[name]
                response = values.pop(0) if isinstance(values, list) else values
            elif name == "SELECT_CATEGORY_TOPK":
                response = json.dumps({"option_ids": [f"C00{i}" for i in range(1, 6)], "summary": "合成排序"})
            elif name == "INFER_CATEGORY_FROM_TOPK":
                response = json.dumps({"option_id": "C001", "summary": "合成选择"})
            else:
                key = {"INTERPRET_FIELD_NAME": "field_semantics", "INTERPRET_DOMAIN": "domain_semantics", "INTERPRET_FIELD_DESCRIPTION": "field_description_semantics", "INTERPRET_TABLE_DESCRIPTION": "table_description_semantics"}[name]
                response = json.dumps({"summary": "合成解释", "data": {key: "已观测的合成语义"}})
        if isinstance(response, Exception):
            raise response
        return response


def run(plans, *, rows=ROWS, overrides=None, **limits):
    module = api()
    generate = Script(plans, overrides)
    controller = module.build_program_controller(generate, REGISTRY, rows, module.ProgramConfig(**limits))
    trace = controller.run({"domain": "合成金融", "field_name": "balance", "source_id": "PRIVATE-ROW-997", "gold_category": "SECRET-GOLD", "metadata": {"field_description": "PRELOADED-SECRET"}})
    return controller, trace, generate


def test_one_plan_executes_whole_program_with_real_guards():
    c, trace, generate = run([program(*FULL)])
    assert type(c.bridge.executor).__name__ == "PoolExecutor"
    assert type(c.bridge.guard).__name__ == "PoolController"
    assert trace.terminated and trace.final_result == {"category": "cat1", "grading_level": "L2"}
    assert generate.kinds == ["planner", "INTERPRET_FIELD_NAME", "SELECT_CATEGORY_TOPK", "INFER_CATEGORY_FROM_TOPK"]
    assert trace.budget["planner_attempts"] == 1 and trace.budget["skill_steps"] == 5
    assert all(s["status"] == "success" and s["binding"]["version"] == "canonical-state-bindings-v1" for s in trace.programs[0]["steps"])
    assert trace.state["_provenance"]["field_semantics"] == "INTERPRET_FIELD_NAME"
    json.dumps(trace.to_dict())


def test_full_catalog_and_only_public_initial_information():
    _, trace, generate = run([program("INTERPRET_DOMAIN")], max_programs=1)
    payload = json.loads(generate.messages[0][1]["content"])
    assert len(payload["skill_library"]) == 14
    assert {"SELECT_CATEGORY_TOPK", "INFER_CATEGORY_FROM_TOPK"} <= {s["name"] for s in payload["skill_library"]}
    assert all(s["purpose"] and s["contract"] and "required_inputs" in s and "terminal" in s for s in payload["skill_library"])
    assert payload["current"]["input"] == {"domain": "合成金融", "field_name": "balance"}
    combined = json.dumps(generate.messages, ensure_ascii=False)
    assert all(secret not in combined for secret in ("PRIVATE-ROW-997", "SECRET-GOLD", "SECRET-GRADE", "PRELOADED-SECRET"))
    assert [s["call"]["tool"] for s in trace.programs[0]["steps"]] == ["INTERPRET_DOMAIN"]
    assert not trace.terminated


def test_replan_only_after_program_end_reuses_state_and_feedback():
    _, trace, generate = run([program("INTERPRET_FIELD_NAME"), program(*FULL[1:])])
    assert trace.terminated and trace.budget["programs"] == 2
    assert generate.kinds[:3] == ["planner", "INTERPRET_FIELD_NAME", "planner"]
    second = json.loads(generate.messages[2][1]["content"])
    assert second["feedback"]["status"] == "completed"
    assert second["feedback"]["plan"]["program"][0]["tool"] == FULL[0]
    assert second["current"]["named_state"]["field_semantics"]
    assert second["feedback"]["remaining_budget"]["skill_steps"] == 19


@pytest.mark.parametrize("raw", ["{}", "[]", "{\"program\":[]}", "```json\n{\"program\":[]}\n```", "{\"program\":[]} {}", "{\"program\":[],\"x\":1}", "{\"program\":[],\"program\":[]}", program("UNKNOWN"), program({"tool": "INTERPRET_FIELD_NAME", "arguments": {"extra": 1}}), program({"tool": "INTERPRET_FIELD_NAME", "arguments": {}, "extra": 1}), program({"tool": "INTERPRET_FIELD_NAME", "arguments": {"field_name": "$state.not_yet"}}), program("FINALIZE_CLASSIFICATION", "INTERPRET_DOMAIN"), program("INFER_GRADING_LEVEL"), program("INFER_CATEGORY_FROM_TOPK"), program({"tool": "INTERPRET_FIELD_NAME", "arguments": {"field_name": "$domain"}})])
def test_invalid_programs_spend_only_planner_attempt(raw):
    _, trace, _ = run([raw], max_planner_attempts=1)
    assert not trace.terminated and trace.budget["skill_steps"] == 0 and trace.budget["programs"] == 0
    assert trace.planner_attempts[0]["status"] == "rejected" and trace.planner_attempts[0]["raw_response"] == raw


def test_future_references_and_automatic_binding():
    _, trace, _ = run([program({"tool": "INTERPRET_FIELD_NAME", "arguments": {"field_name": "$metadata.field_name"}}, {"tool": "SELECT_CATEGORY_TOPK", "arguments": {"semantic_query": "$state.semantic_query"}}, *FULL[2:])])
    assert trace.terminated


def test_static_length_limit_is_independent():
    _, trace, _ = run([program(*FULL)], max_program_steps=4, max_planner_attempts=1)
    assert trace.budget["skill_steps"] == 0 and "length" in trace.planner_attempts[0]["error"]


def test_total_skill_budget_does_not_reset_or_inject_finalizer():
    _, trace, _ = run([program("INTERPRET_FIELD_NAME"), program(*FULL[1:])], max_skill_steps=3)
    assert trace.budget["skill_steps"] == 3 and trace.stop_reason == "skill_budget_exhausted"
    assert not trace.terminated and "grading_level" not in trace.state
    assert trace.programs[1]["status"] == "budget_exhausted"


def test_attempt_budget_survives_invalid_generation_and_generator_exception():
    _, trace, _ = run(["not JSON", RuntimeError("temporary model failure"), program("INTERPRET_DOMAIN")], max_planner_attempts=3)
    assert trace.budget["planner_attempts"] == 3 and trace.budget["programs"] == 1
    assert [a["status"] for a in trace.planner_attempts] == ["rejected", "error", "accepted"]
    assert trace.stop_reason == "planner_budget_exhausted"


def test_missing_metadata_remains_unknown_after_batch_and_fails_at_runtime():
    rows = [{"id": "PRIVATE-ROW-997", "domain": "合成金融", "metadata": {"field_name": "balance"}}]
    _, trace, _ = run([program("GET_FIELD_METADATA", "INTERPRET_FIELD_DESCRIPTION"), program(*FULL)], rows=rows)
    assert trace.terminated
    assert trace.programs[0]["status"] == "failed" and trace.programs[0]["failure_index"] == 1
    assert trace.programs[0]["steps"][1]["status"] == "blocked"
    assert not any(e["type"] == "field_description" for e in trace.state["availability_ledger"])
    assert "field_description" not in trace.state


def test_scoped_unavailable_blocks_repeat_but_errors_remain_retryable():
    module = api()
    from pool_availability import initialize, observe, blocked, status
    from agent.orchestration.contracts import MetadataRecord, ToolCall
    from agent.orchestration.runtime import Observation
    record = MetadataRecord.from_mapping({"domain": "合成金融", "field_name": "balance"})
    state = {}; initialize(record, state)
    call = ToolCall("GET_FIELD_DESCRIPTION", {})
    observe(call, Observation(call.tool, "error", "temporary", {}), state)
    assert status(state, "record", "balance", "field_description") == "error" and not blocked(call, state)
    observe(call, Observation(call.tool, "unavailable", "missing", {}, {"source": "canonical-val", "missing": ["field_description"]}), state)
    assert blocked(call, state)
    # Bridge must use exactly this production availability module.
    c = module.build_program_controller(Script([]), REGISTRY, ROWS)
    assert c.bridge.availability is importlib.import_module("pool_availability")


def test_duplicate_acquisition_is_counted_and_ends_program():
    _, trace, _ = run([program("GET_FIELD_METADATA"), program("GET_FIELD_METADATA", "INTERPRET_DOMAIN")], max_programs=2)
    assert trace.budget["skill_steps"] == 2
    assert trace.programs[1]["steps"][0]["status"] == "blocked" and len(trace.programs[1]["steps"]) == 1


def test_semantic_error_ends_program_then_retries_without_losing_raw_response():
    _, trace, generate = run([program("INTERPRET_FIELD_NAME", "SELECT_CATEGORY_TOPK"), program(*FULL)], overrides={"INTERPRET_FIELD_NAME": ["{\"summary\":\"bad\",\"data\":{\"field_semantics\":{}}}", json.dumps({"summary": "retry", "data": {"field_semantics": "recovered"}})]})
    assert trace.terminated and trace.budget["skill_steps"] == 6
    assert generate.kinds[:4] == ["planner", "INTERPRET_FIELD_NAME", "planner", "INTERPRET_FIELD_NAME"]
    assert trace.programs[0]["steps"][0]["metadata"]["raw_response"]


def test_outside_actual_topk_selection_fails_real_executor_and_is_retained():
    bad = json.dumps({"option_id": "C006", "summary": "outside"})
    _, trace, _ = run([program(*FULL)], max_programs=1, overrides={"INFER_CATEGORY_FROM_TOPK": [bad, bad]})
    step = trace.programs[0]["steps"][-1]
    assert step["status"] == "error" and step["call"]["tool"] == "INFER_CATEGORY_FROM_TOPK"
    assert len(step["metadata"]["attempts"]) == 2 and not trace.terminated


@pytest.mark.parametrize("tool,argument,value", [("INTERPRET_FIELD_DESCRIPTION", "field_description", "invented raw"), ("INFER_CATEGORY_FROM_TOPK", "candidate_standard_categories", [{"category": "cat6"}]), ("INFER_GRADING_LEVEL", "aligned_category", "cat6"), ("FINALIZE_CLASSIFICATION", "final_evidence_state", {"category": "cat1", "grading_level": "L4"})])
def test_literal_forgery_never_overrides_actual_state(tool, argument, value):
    setup = ["GET_FIELD_METADATA", *FULL[:-1]]
    _, trace, _ = run([program(*setup, {"tool": tool, "arguments": {argument: value}})], max_programs=1, max_planner_attempts=1)
    assert not trace.terminated
    assert trace.planner_attempts[0]["status"] == "rejected"
    assert argument in trace.planner_attempts[0]["error"]
    assert trace.budget == {"programs": 0, "planner_attempts": 1, "skill_steps": 0}


@pytest.mark.parametrize("changed_tool", ["SELECT_CATEGORY_TOPK", "INTERPRET_DOMAIN", "GET_FIELD_METADATA"])
def test_changed_evidence_or_candidates_invalidates_old_selection_and_grade(changed_tool):
    plans = [program(*FULL[:-1]), program(changed_tool), program("FINALIZE_CLASSIFICATION")]
    overrides = {"SELECT_CATEGORY_TOPK": [json.dumps({"option_ids": [f"C00{i}" for i in range(1, 6)], "summary": "first"}), json.dumps({"option_ids": [f"C00{i}" for i in range(2, 7)], "summary": "new"})]}
    _, trace, _ = run(plans, overrides=overrides, max_planner_attempts=3)
    assert not trace.terminated and "category" not in trace.state and "grading_level" not in trace.state and "final_evidence_state" not in trace.state
    assert trace.planner_attempts[-1]["status"] == "rejected"
    assert trace.programs[1]["invalidations"]


def test_model_error_does_not_publish_or_reuse_unverified_evidence():
    _, trace, _ = run([program("INTERPRET_DOMAIN"), program("INTERPRET_DOMAIN")], max_programs=2, overrides={"INTERPRET_DOMAIN": [RuntimeError("transient"), json.dumps({"summary": "ok", "data": {"domain_semantics": "valid"}})]})
    assert trace.programs[0]["steps"][0]["status"] == "error"
    assert trace.programs[1]["steps"][0]["status"] == "success"


def test_trace_export_refuses_overwrite(tmp_path):
    _, trace, _ = run([program(*FULL)])
    target = tmp_path / "trace.json"
    trace.write_json(target)
    original = target.read_bytes()
    with pytest.raises(FileExistsError): trace.write_json(target)
    assert target.read_bytes() == original


def test_cli_help_demo_and_missing_local_model(tmp_path):
    api()
    help_result = subprocess.run([sys.executable, "-m", "program_loop", "--help"], capture_output=True, text=True)
    assert help_result.returncode == 0 and "--output" in help_result.stdout
    target = tmp_path / "demo.json"
    demo = subprocess.run([sys.executable, "-m", "program_loop", "--output", str(target)], capture_output=True, text=True)
    assert demo.returncode == 0 and "scripted" in demo.stdout
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["terminated"] and payload["execution_mode"] == "scripted model I/O + real executor"
    repeated = subprocess.run([sys.executable, "-m", "program_loop", "--output", str(target)], capture_output=True, text=True)
    assert repeated.returncode != 0
    missing = subprocess.run([sys.executable, "-m", "program_loop", "--local-model", str(tmp_path / "missing")], capture_output=True, text=True)
    assert missing.returncode != 0 and "local model path" in missing.stderr


def test_default_import_does_not_import_torch():
    api()
    result = subprocess.run([sys.executable, "-c", "import program_loop,sys; assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_publication_failure_is_transactional_and_retains_model_response(monkeypatch):
    api()
    import program_loop.execution as bridge_module

    def failed_publication(record, state, data, skill):
        state["injected_unverified"] = "must not escape"
        raise RuntimeError("publication interrupted")

    monkeypatch.setattr(bridge_module, "publish_state", failed_publication)
    _, trace, _ = run([program("INTERPRET_DOMAIN")], max_programs=1)
    step = trace.programs[0]["steps"][0]
    assert step["status"] == "error" and step["data"] == {}
    assert "domain_semantics" not in trace.state and "injected_unverified" not in trace.state
    assert step["metadata"]["raw_response"]
    assert "publication interrupted" in step["summary"]


@pytest.mark.parametrize("response", [b"not text", float("nan")])
def test_nontext_planner_response_is_rejected_with_serializable_diagnostic(response):
    _, trace, _ = run([response], max_planner_attempts=1)
    assert trace.planner_attempts[0]["status"] == "rejected"
    assert trace.budget == {"programs": 0, "planner_attempts": 1, "skill_steps": 0}
    json.dumps(trace.to_dict(), allow_nan=False)


def test_real_source_missing_and_lookup_error_have_distinct_retry_semantics():
    _, missing, _ = run([program("GET_FIELD_METADATA", "INTERPRET_DOMAIN"), program("GET_FIELD_METADATA")], rows=[], max_programs=2)
    assert [p["steps"][0]["status"] for p in missing.programs] == ["error", "error"]
    assert missing.state["availability_ledger"][-1]["status"] == "error"
    assert "field_metadata" not in missing.state
    rows = [{"id": "PRIVATE-ROW-997", "domain": "合成金融", "metadata": {"field_name": "balance"}}]
    _, absent, _ = run([program("GET_FIELD_METADATA", "INTERPRET_FIELD_DESCRIPTION"), program("GET_FIELD_METADATA")], rows=rows, max_programs=2)
    assert absent.programs[0]["steps"][0]["status"] == "success"
    assert absent.programs[0]["steps"][1]["status"] == "blocked"
    assert absent.programs[1]["steps"][0]["status"] == "blocked"
    assert absent.budget["skill_steps"] == 3
    assert not any(e["status"] == "unavailable" for e in absent.state["availability_ledger"])


def test_fresh_evidence_can_reselect_and_finalize_in_next_program():
    _, trace, _ = run([program(*FULL[:-1]), program("INTERPRET_DOMAIN"), program(*FULL[1:])])
    assert trace.terminated and trace.budget["programs"] == 3
    assert {"category", "grading_level", "final_evidence_state"} <= set(trace.programs[1]["invalidations"])
    assert trace.programs[2]["steps"][0]["lineage"]["evidence"] == 2


def test_static_future_reference_rejects_stale_grade_after_evidence_step():
    _, trace, _ = run([program(*FULL[:-1]), program("INTERPRET_DOMAIN", "FINALIZE_CLASSIFICATION")], max_planner_attempts=2)
    assert trace.planner_attempts[-1]["status"] == "rejected"
    assert trace.budget["skill_steps"] == 4
    assert "category" in trace.state  # Static rejection leaves prior verified evidence intact.


def test_no_name_interpretation_step_is_required_for_legal_terminal():
    _, trace, generate = run([program("GET_FIELD_METADATA", *FULL[1:])])
    assert trace.terminated and "INTERPRET_FIELD_NAME" not in generate.kinds
    assert "field_semantics" not in trace.state


def test_private_identity_and_labels_stay_out_of_acquisition_replanning_messages():
    _, trace, generate = run([program("GET_FIELD_METADATA"), program(*FULL[1:])])
    assert trace.terminated
    combined = json.dumps(generate.messages, ensure_ascii=False)
    assert "当前余额" in combined
    assert all(secret not in combined for secret in ("PRIVATE-ROW-997", "SECRET-GOLD", "SECRET-GRADE", "PRELOADED-SECRET"))


def test_real_terminal_guard_is_used_and_rejection_does_not_publish(monkeypatch):
    module = api()
    script = Script([program(*FULL)])
    controller = module.build_program_controller(script, REGISTRY, ROWS, module.ProgramConfig(max_programs=1))
    original = controller.bridge.guard._valid_terminal
    checked = []

    def reject_after_real_validation(result, state):
        checked.append(original(result, state))
        return False

    monkeypatch.setattr(controller.bridge.guard, "_valid_terminal", reject_after_real_validation)
    trace = controller.run({"domain": "合成金融", "field_name": "balance", "source_id": "PRIVATE-ROW-997"})
    assert checked == [True]
    assert not trace.terminated and "final_result" not in trace.state
    assert trace.programs[0]["steps"][-1]["status"] == "error"


def test_nontext_skill_response_is_retained_without_breaking_feedback():
    _, trace, generate = run([program("INTERPRET_DOMAIN"), program(*FULL)], overrides={"INTERPRET_DOMAIN": b"invalid model output"})
    assert trace.terminated
    assert trace.programs[0]["steps"][0]["status"] == "error"
    assert "invalid model output" in json.dumps(trace.to_dict(), allow_nan=False)
    assert generate.kinds[:3] == ["planner", "INTERPRET_DOMAIN", "planner"]


def test_raw_metadata_acquisition_does_not_statically_claim_semantic_evidence():
    _, trace, _ = run([program("GET_FIELD_METADATA", "SELECT_CATEGORY_TOPK", "COMPARE_STANDARD_CANDIDATES")], max_planner_attempts=1)
    assert trace.planner_attempts[0]["status"] == "rejected"
    assert "semantic_evidence" in trace.planner_attempts[0]["error"]
    assert trace.budget["skill_steps"] == 0


def comparison_raw(entry, text="候选比较依据为已观测语义"):
    return json.dumps({"summary": "比较完成", "data": {"standard_comparison": text, "aligned_standard_entry": entry}}, ensure_ascii=False)


@pytest.mark.parametrize("full", [False, True])
def test_comparison_binds_authoritative_entry_and_continues_to_finalize(full):
    entry = {"category": "cat1", **REGISTRY["cat1"]}
    if not full:
        entry.pop("grading_level")
    raw = comparison_raw(entry)
    _, trace, generate = run([program(*FULL[:2], "COMPARE_STANDARD_CANDIDATES", *FULL[2:])],
                            overrides={"COMPARE_STANDARD_CANDIDATES": raw}, max_programs=1)
    assert trace.terminated, trace.to_dict()
    step = trace.programs[0]["steps"][2]
    assert step["metadata"]["raw_response"] == raw
    assert trace.state["aligned_standard_entry"] == {"category": "cat1", **REGISTRY["cat1"]}
    messages = next(m for m in generate.messages if "selected Skill COMPARE_STANDARD_CANDIDATES." in m[0]["content"])
    payload = json.loads(messages[1]["content"])
    assert "non-empty string" in payload["output_schema"]["data"]["standard_comparison"]
    assert "grading_level" not in json.dumps(messages, ensure_ascii=False)


@pytest.mark.parametrize("change", ["list", "empty", "outside", "label", "description", "grade", "extra", "partial", "conflict"])
def test_comparison_rejects_invalid_text_or_candidate_without_publication(change):
    entry = {"category": "cat1", "label": REGISTRY["cat1"]["label"], "description": REGISTRY["cat1"]["description"]}
    text = "比较依据"
    if change == "list": text = ["比较依据"]
    elif change == "empty": text = "  "
    elif change == "outside": entry = {"category": "cat6", **REGISTRY["cat6"]}
    elif change in ("label", "description"): entry[change] = "forged"
    elif change == "grade": entry["grading_level"] = "L999"
    elif change == "extra": entry["extra"] = "forged"
    elif change == "partial": entry.pop("description")
    elif change == "conflict": entry = {"category": "cat2", **REGISTRY["cat2"]}
    raw = comparison_raw(entry, text)
    tools = (*FULL[:3], "COMPARE_STANDARD_CANDIDATES") if change == "conflict" else (*FULL[:2], "COMPARE_STANDARD_CANDIDATES")
    _, trace, _ = run([program(*tools)], overrides={"COMPARE_STANDARD_CANDIDATES": raw}, max_programs=1)
    step = trace.programs[0]["steps"][-1]
    assert step["status"] == "error"
    assert step["metadata"]["raw_response"] == raw
    assert "standard_comparison" not in trace.state and not trace.terminated
