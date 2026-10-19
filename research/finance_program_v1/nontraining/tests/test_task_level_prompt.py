"""Prompt contract and CPU boundaries; scripted I/O is not model efficacy evidence."""
import json

import pytest

from test_program_loop import FULL, program, run


OBJECTIVE = "Plan the complete remaining classification task using verified evidence and frozen Registry grading, through legal terminal execution."
INSTRUCTIONS = (
    "Plan the complete remaining classification task in one ordered Program, not just the next Skill.",
    "Choose the Skills and their order yourself from the complete P14 library.",
    "Plan all dependencies through a successful terminal; selecting candidates alone does not finish classification.",
    "You need not see a future output now: use its documented canonical reference after its producer; execution will bind the actual verified output.",
    "Conditional outputs may be missing or execution may fail; the Program then stops and its factual observation returns to you at the outer boundary.",
    "The parser accepts partial Programs, but that is a permissive syntax boundary, not the planning objective.",
)
PAYLOAD_KEYS = {"task", "current", "skill_library", "remaining_budget", "feedback", "recovery_context", "availability_rule"}


def planner_payloads(script):
    payloads = []
    for messages in script.messages:
        if "Global Program Planner" not in messages[0]["content"]:
            continue
        assert [message["role"] for message in messages] == ["system", "user"]
        assert all(set(message) == {"role", "content"} for message in messages)
        system = messages[0]["content"]
        assert all(instruction in system for instruction in INSTRUCTIONS)
        assert "No Skill name is mandatory and the controller inserts no steps." in system
        assert "Programs may be incomplete." not in system
        assert '{"program":[{"tool":"SKILL_NAME","arguments":{}}]}' not in system
        payload = json.loads(messages[1]["content"])
        assert set(payload) == PAYLOAD_KEYS
        assert payload["task"] == OBJECTIVE
        assert len(payload["skill_library"]) == 14
        payloads.append(payload)
    combined = json.dumps(script.messages, ensure_ascii=False)
    assert all(secret not in combined for secret in ("PRIVATE-ROW-997", "SECRET-GOLD", "SECRET-GRADE", "PRELOADED-SECRET"))
    return payloads


def test_planner_requests_complete_remaining_task_instead_of_one_next_action():
    _, _, script = run([program("INTERPRET_DOMAIN")], max_programs=1, max_planner_attempts=4)
    payload = planner_payloads(script)[0]
    assert payload["current"]["input"] == {"domain": "合成金融", "field_name": "balance"}
    assert payload["feedback"] is None
    assert payload["recovery_context"]["previous_program"] is None
    assert not payload["recovery_context"]["classification_completed"]
    assert payload["remaining_budget"] == {"programs": 1, "planner_attempts": 4, "skill_steps": 20, "program_steps": 10}


def test_legal_partial_program_is_not_rejected_or_completed_by_controller():
    _, trace, _ = run([program("INTERPRET_DOMAIN")], max_programs=1)
    assert trace.planner_attempts[0]["status"] == "accepted"
    assert trace.budget == {"programs": 1, "planner_attempts": 1, "skill_steps": 1}
    assert not trace.terminated and trace.final_result is None
    assert trace.stop_reason == "program_budget_exhausted"
    assert [step["call"]["tool"] for step in trace.programs[0]["steps"]] == ["INTERPRET_DOMAIN"]


def test_complete_program_still_uses_real_terminal_guard():
    controller, trace, script = run([program(*FULL)])
    assert type(controller.bridge.guard).__name__ == "PoolController"
    assert trace.terminated and trace.stop_reason == "validated_terminal"
    assert trace.final_result == {"category": "cat1", "grading_level": "L2"}
    assert trace.programs[0]["steps"][-1]["status"] == "success"
    assert trace.budget == {"programs": 1, "planner_attempts": 1, "skill_steps": 5}
    assert script.kinds.count("planner") == 1
    assert len(planner_payloads(script)) == 1


@pytest.mark.parametrize("first,status,remaining,used", [
    (program("INTERPRET_FIELD_NAME"), "completed", {"programs": 1, "planner_attempts": 1, "skill_steps": 11, "program_steps": 8}, {"programs": 2, "planner_attempts": 2, "skill_steps": 2}),
    (program("GET_FIELD_METADATA", "INTERPRET_FIELD_DESCRIPTION", "INTERPRET_DOMAIN"), "failed", {"programs": 1, "planner_attempts": 1, "skill_steps": 10, "program_steps": 8}, {"programs": 2, "planner_attempts": 2, "skill_steps": 3}),
    ("not JSON", "planner_rejected", {"programs": 2, "planner_attempts": 1, "skill_steps": 12, "program_steps": 8}, {"programs": 1, "planner_attempts": 2, "skill_steps": 1}),
])
def test_task_objective_preserves_outer_facts_and_budget(first, status, remaining, used):
    rows = [{"id": "PRIVATE-ROW-997", "domain": "合成金融", "metadata": {"field_name": "balance"}}]
    _, trace, script = run([first, program("INTERPRET_DOMAIN")], rows=rows,
        max_programs=2, max_planner_attempts=2, max_skill_steps=12, max_program_steps=8)
    payloads = planner_payloads(script)
    assert len(payloads) == 2 and script.kinds.count("planner") == 2
    second = payloads[1]
    assert second["feedback"]["status"] == status
    assert second["feedback"]["remaining_budget"] == remaining
    assert second["remaining_budget"] == second["recovery_context"]["remaining_budget"] == remaining
    assert not second["recovery_context"]["classification_completed"]
    assert trace.budget == used and not trace.terminated and trace.final_result is None
    if status == "planner_rejected":
        assert second["recovery_context"]["previous_program"] is None
        assert trace.planner_attempts[0]["raw_response"] == first
        assert second["feedback"]["error"] == trace.planner_attempts[0]["error"]
    else:
        assert second["feedback"] == trace.programs[0]
        previous = second["recovery_context"]["previous_program"]
        assert previous["status"] == status
        if status == "completed":
            assert second["current"]["named_state"]["field_semantics"]
            assert previous["successful_steps"] == [{"step_index": 0, "tool": "INTERPRET_FIELD_NAME"}]
            assert previous["unexecuted_tail"] == []
        else:
            assert previous["failure_index"] == 1
            assert previous["failed_steps"] == [{"step_index": 1, "tool": "INTERPRET_FIELD_DESCRIPTION", "status": "blocked"}]
            assert previous["unexecuted_tail"] == [{"step_index": 2, "tool": "INTERPRET_DOMAIN"}]
            assert "field_description" not in second["current"]["named_state"]


def test_hard_budget_prompt_keeps_nine_step_raw_program_rejected_at_eight_step_gate():
    raw = program("INTERPRET_DOMAIN", "GET_FIELD_METADATA", "INTERPRET_FIELD_DESCRIPTION",
                  "INTERPRET_TABLE_DESCRIPTION", "INTERPRET_FIELD_NAME", *FULL[1:])
    assert len(json.loads(raw)["program"]) == 9
    _, trace, script = run([raw], max_programs=3, max_program_steps=8, max_skill_steps=12, max_planner_attempts=1)
    payload = planner_payloads(script)[0]
    assert payload["remaining_budget"] == {"programs": 3, "planner_attempts": 1, "skill_steps": 12, "program_steps": 8}
    assert trace.budget == {"programs": 0, "planner_attempts": 1, "skill_steps": 0}
    assert trace.programs == [] and script.kinds == ["planner"]
    assert len(trace.planner_attempts) == 1
    assert trace.planner_attempts[0]["status"] == "rejected"
    assert trace.planner_attempts[0]["raw_response"] == raw
    assert "length" in trace.planner_attempts[0]["error"]
    assert not trace.terminated and trace.final_result is None
    assert trace.stop_reason == "planner_budget_exhausted"
    instructions = (
        "Each Program must fit remaining_budget.program_steps and remaining_budget.skill_steps; these are hard upper bounds, not suggested plan lengths.",
        "Use a sufficient dependency path for the remaining task; complete classification does not require using every Skill or every possible evidence source.",
        "Before returning JSON, count the planned steps and omit nonessential operations if it exceeds either bound, without omitting required dependencies or inventing evidence.",
    )
    system = script.messages[0][0]["content"]
    assert INSTRUCTIONS[-1] + "\n" + "\n".join(instructions) + "\n" in system


def test_dependency_alternatives_prompt_preserves_five_step_terminal_and_requires_all_boundary():
    limits = {"max_programs": 3, "max_program_steps": 8, "max_skill_steps": 12, "max_planner_attempts": 4}
    controller, trace, script = run([program(*FULL)], **limits)
    assert type(controller.bridge.executor).__name__ == "PoolExecutor"
    assert type(controller.bridge.guard).__name__ == "PoolController"
    assert trace.terminated and trace.stop_reason == "validated_terminal"
    assert trace.final_result == {"category": "cat1", "grading_level": "L2"}
    assert trace.budget == {"programs": 1, "planner_attempts": 1, "skill_steps": 5}
    assert script.kinds.count("planner") == 1
    assert len(trace.programs) == 1 and len(trace.programs[0]["steps"]) == 5
    assert tuple(step["call"]["tool"] for step in trace.programs[0]["steps"]) == FULL
    assert all(step["status"] == "success" for step in trace.programs[0]["steps"])
    payload = planner_payloads(script)[0]
    assert payload["remaining_budget"] == {"programs": 3, "planner_attempts": 4, "skill_steps": 12, "program_steps": 8}

    incomplete = tuple(tool for tool in FULL if tool != "INFER_GRADING_LEVEL")
    raw = program(*incomplete)
    assert len(json.loads(raw)["program"]) == 4
    _, rejected, rejected_script = run([raw], **{**limits, "max_planner_attempts": 1})
    assert rejected.budget == {"programs": 0, "planner_attempts": 1, "skill_steps": 0}
    assert rejected.programs == [] and rejected_script.kinds == ["planner"]
    assert len(rejected.planner_attempts) == 1
    assert rejected.planner_attempts[0]["status"] == "rejected"
    assert rejected.planner_attempts[0]["raw_response"] == raw
    assert "final_evidence_state" in rejected.planner_attempts[0]["error"]
    assert not rejected.terminated and rejected.final_result is None
    assert rejected.stop_reason == "planner_budget_exhausted"
    assert planner_payloads(rejected_script)[0]["remaining_budget"] == {"programs": 3, "planner_attempts": 1, "skill_steps": 12, "program_steps": 8}

    instructions = (
        "Each possible_producers list contains alternative ways to supply that input, not a sequence that must all be executed; choose a sufficient dependency path and still verify its conditional outputs at execution.",
        "all_available_evidence means the verified evidence actually accumulated by the chosen Skills, not every possible evidence source; only a nonempty requires_all list requires every named prerequisite.",
    )
    anchor = "Before returning JSON, count the planned steps and omit nonessential operations if it exceeds either bound, without omitting required dependencies or inventing evidence."
    for generated in (script, rejected_script):
        system = generated.messages[0][0]["content"]
        assert anchor + "\n" + "\n".join(instructions) + "\n" in system
