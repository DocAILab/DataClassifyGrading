"""Real F66 controller, binder and guards; every provider reply is synthetic."""
import json
from copy import deepcopy
import pytest
from program_loop import ProgramConfig, build_program_controller
from program_loop.prompt_encoding import unpack_prompt

REGISTRY = {f"cat{i}": {"label": f"Synthetic {i}", "description": f"Public definition {i}", "grading_level": "L2"}
            for i in range(1, 7)}
ROWS = [{"id": "PRIVATE-CPU-SOURCE", "domain": "finance", "metadata": {
    "field_name": "CPU_FIELD", "field_description": "显式合成描述,保持原标点", "table_name": "CPU_TABLE"}}]
FULL = ["GET_FIELD_METADATA", "INTERPRET_FIELD_NAME", "SEARCH_STANDARD_BY_FIELD", "INFER_CATEGORY", "INFER_GRADING_LEVEL", "FINALIZE_CLASSIFICATION"]


def program(*names):
    return json.dumps({"program": [{"tool": name, "arguments": {}} for name in names]})


class SyntheticProvider:
    def __init__(self, plans, *, initial_failure=False, insufficient=False):
        self.plans = iter(plans)
        self.initial_failure = initial_failure
        self.insufficient = insufficient
        self.planners = []
        self.calls = []
        self.fusions = 0

    def __call__(self, messages):
        self.calls.append(deepcopy(messages))
        system = messages[0]["content"]
        if "Global Program Planner" in system:
            self.planners.append(unpack_prompt(json.loads(messages[1]["content"])))
            return next(self.plans)
        stage = system.split("selected Skill ")[1].split(".")[0]
        payload = json.loads(messages[1]["content"])
        if stage == "INTERPRET_FIELD_NAME":
            if self.initial_failure:
                self.initial_failure = False
                return "ORIGINAL_SYNTHETIC_FAILURE_MARKER"
            context = payload["grounding_context"]
            value = {"schema": "grounded-source-v1", "status": "observed_description" if context.get("field_description") else "unknown",
                     "field_name": context["field_name"], "description": context.get("field_description"), "source_consistency": "not_adjudicated"}
            return json.dumps({"summary": value["status"], "data": {"field_semantics": value}})
        if stage in {"SEARCH_STANDARD_BY_FIELD", "SELECT_CATEGORY_TOPK"}:
            return json.dumps({"option_ids": [f"C00{i}" for i in range(1, 6)], "summary": "synthetic ranking"})
        if stage in {"INFER_CATEGORY", "INFER_CATEGORY_FROM_TOPK"}:
            return json.dumps({"option_id": None if self.insufficient else "C001", "summary": "actual subject evidence missing" if self.insufficient else "synthetic selection"})
        if stage == "INTERPRET_DOMAIN":
            return json.dumps({"summary": "synthetic", "data": {"domain_semantics": "finance"}})
        if stage == "FUSE_FIELD_EVIDENCE":
            self.fusions += 1
            return json.dumps({"summary": "synthetic", "data": {"field_evidence": "same source facts; model wording " + str(self.fusions)}})
        raise AssertionError("unplanned synthetic stage: " + stage)


def run(provider, **limits):
    cfg = dict(max_programs=3, max_program_steps=10, max_skill_steps=12, max_planner_attempts=4)
    cfg.update(limits)
    controller = build_program_controller(provider, REGISTRY, ROWS, ProgramConfig(**cfg), skill_pool="F66")
    trace = controller.run({"domain": "finance", "field_name": "CPU_FIELD", "source_id": "PRIVATE-CPU-SOURCE"})
    return controller, trace


@pytest.mark.parametrize("middle", ["parser_rejection", "successful_partial"])
def test_actual_failure_survives_intervening_observation(middle):
    provider = SyntheticProvider([program("INTERPRET_FIELD_NAME"), "{}" if middle == "parser_rejection" else program("GET_FIELD_METADATA"), program("GET_FIELD_METADATA")], initial_failure=True)
    _, trace = run(provider, max_planner_attempts=3)
    memory = provider.planners[2]["recovery_context"].get("failure_history", [])
    assert any(item["tool"] == "INTERPRET_FIELD_NAME" for item in memory)
    assert trace.programs[0]["steps"][0]["status"] == "error"
    assert "PRIVATE-CPU-SOURCE" not in json.dumps(provider.planners)


def test_plan_longer_than_total_remaining_skill_budget_is_rejected_before_execution():
    provider = SyntheticProvider([program("INTERPRET_DOMAIN", "INTERPRET_FIELD_NAME")])
    _, trace = run(provider, max_skill_steps=1, max_planner_attempts=1)
    assert trace.budget["programs"] == trace.budget["skill_steps"] == 0
    assert trace.planner_attempts[0]["status"] == "rejected"


def test_repeated_model_rewording_does_not_count_as_new_source_progress():
    provider = SyntheticProvider([program("GET_FIELD_METADATA", "INTERPRET_FIELD_NAME", "FUSE_FIELD_EVIDENCE")] + [program("FUSE_FIELD_EVIDENCE")] * 7)
    _, trace = run(provider, max_programs=8, max_planner_attempts=8, max_skill_steps=20)
    assert trace.stop_reason == "no_progress"
    assert len(provider.planners) <= 4
    assert not trace.terminated and trace.final_result is None


def test_normal_legal_program_still_completes_with_original_guards():
    provider = SyntheticProvider([program(*FULL)])
    _, trace = run(provider)
    assert trace.terminated and trace.stop_reason == "validated_terminal"
    assert trace.final_result == {"category": "cat1", "grading_level": "L2"}


def test_failure_history_is_fresh_for_each_source_rollout():
    provider = SyntheticProvider([program("INTERPRET_FIELD_NAME"), program("GET_FIELD_METADATA")], initial_failure=True)
    controller, first = run(provider, max_programs=2, max_planner_attempts=2)
    assert first.programs[0]["steps"][0]["status"] == "error"
    provider.plans = iter([program(*FULL)])
    controller.config = ProgramConfig(max_programs=1, max_planner_attempts=1)
    trace = controller.run({"domain": "finance", "field_name": "CPU_FIELD", "source_id": "PRIVATE-CPU-SOURCE"})
    assert provider.planners[-1]["recovery_context"].get("failure_history") == []
    assert trace.terminated


def test_unknown_type_is_never_declared_globally_absent():
    provider = SyntheticProvider([program("GET_FIELD_METADATA"), program("INTERPRET_FIELD_NAME")])
    _, trace = run(provider, max_programs=2, max_planner_attempts=2)
    readiness = provider.planners[1]["recovery_context"].get("input_readiness", {})
    assert readiness.get("metadata_inputs", {}).get("field_type", {}).get("status") == "unknown"
    assert readiness["new_business_facts"] is False
