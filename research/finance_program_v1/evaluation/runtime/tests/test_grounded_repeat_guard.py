"""Reproduce unchanged effective requests becoming callable after unrelated lineage changes."""

import json
from program_loop import ProgramConfig, build_program_controller


REGISTRY = {
    f"cat{i}": {
        "label": f"Synthetic {i}",
        "description": f"Definition {i}",
        "grading_level": "L2",
    }
    for i in range(1, 7)
}
ROWS = [
    {
        "id": "CPU-REPEAT-SOURCE",
        "domain": "finance",
        "metadata": {"field_name": "CPU_FIELD", "field_description": "显式合成,原标点"},
    }
]


class Provider:
    def __init__(self, plans, *, invalid=False):
        self.plans = iter(plans)
        self.grounded = []
        self.invalid = invalid

    def __call__(self, messages):
        if "Global Program Planner" in messages[0]["content"]:
            return json.dumps(
                {
                    "program": [
                        {"tool": name, "arguments": {}} for name in next(self.plans)
                    ]
                }
            )
        payload = json.loads(messages[1]["content"])
        if payload.get("grounding_protocol") == "extractive-grounded-source-v1":
            self.grounded.append(messages)
            if self.invalid:
                return "SYNTHETIC_INVALID_JSON"
            context = payload["grounding_context"]
            observed = bool(context.get("field_description"))
            return json.dumps(
                {
                    "summary": "observed_description" if observed else "unknown",
                    "data": {
                        "field_semantics": {
                            "schema": "grounded-source-v1",
                            "status": "observed_description" if observed else "unknown",
                            "field_name": context["field_name"],
                            "description": context.get("field_description"),
                            "source_consistency": "not_adjudicated",
                        }
                    },
                }
            )
        return json.dumps(
            {"summary": "synthetic domain", "data": {"domain_semantics": "finance"}}
        )


def execute(provider):
    controller = build_program_controller(
        provider,
        REGISTRY,
        ROWS,
        ProgramConfig(max_programs=3, max_planner_attempts=3, max_skill_steps=12),
        skill_pool="F66",
    )
    trace = controller.run(
        {
            "source_id": "CPU-REPEAT-SOURCE",
            "domain": "finance",
            "field_name": "CPU_FIELD",
        }
    )
    return controller, trace


def test_unrelated_semantic_change_does_not_reenable_identical_grounded_model_request():
    provider = Provider(
        [
            ["GET_FIELD_METADATA", "INTERPRET_FIELD_NAME", "INTERPRET_DOMAIN"],
            ["INTERPRET_FIELD_NAME"],
            ["GET_FIELD_METADATA"],
        ]
    )
    _, trace = execute(provider)
    assert len(provider.grounded) == 1
    assert any(
        "identical grounded model request" in s.get("summary", "")
        for p in trace.programs
        for s in p["steps"]
    )
    assert trace.budget["skill_steps"] >= 4  # a blocked native Skill is still metered


def test_returned_invalid_json_is_not_regenerated_for_the_same_unchanged_request():
    provider = Provider(
        [
            ["GET_FIELD_METADATA", "INTERPRET_FIELD_NAME"],
            ["INTERPRET_FIELD_NAME"],
            ["INTERPRET_FIELD_NAME"],
        ],
        invalid=True,
    )
    _, trace = execute(provider)
    assert len(provider.grounded) == 1
    assert not trace.terminated
    assert (
        sum(s["status"] == "success" for p in trace.programs for s in p["steps"]) == 1
    )


def test_real_newly_acquired_description_allows_a_new_request_and_each_source_run_resets_guard():
    plans = [
        ["INTERPRET_FIELD_NAME"],
        ["GET_FIELD_METADATA", "INTERPRET_FIELD_NAME"],
        ["GET_FIELD_METADATA"],
    ]
    provider = Provider(plans)
    controller, trace = execute(provider)
    assert len(provider.grounded) == 2
    assert provider.grounded[0] != provider.grounded[1]
    provider.plans = iter(plans)
    controller.run(
        {
            "source_id": "CPU-REPEAT-SOURCE",
            "domain": "finance",
            "field_name": "CPU_FIELD",
        }
    )
    assert len(provider.grounded) == 4


def test_blocked_request_does_not_inherit_previous_model_token_or_eos_metadata():
    import pytest
    from program_loop.grounded_request_guard import GroundedRequestGuard

    class Metered:
        last_generation_metadata = {
            "token_usage": {"prompt_tokens": 17, "completion_tokens": 4},
            "ended_with_eos": True,
        }

        def __call__(self, messages):
            return "explicit synthetic response"

    guard = GroundedRequestGuard(Metered(), 1)
    messages = [
        {"role": "system", "content": "explicit synthetic contract"},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "skill": "INTERPRET_FIELD_NAME",
                    "grounding_protocol": "extractive-grounded-source-v1",
                }
            ),
        },
    ]
    guard(messages)
    with pytest.raises(ValueError):
        guard(messages)
    assert guard.last_generation_metadata["model_call_count"] == 0
    assert guard.last_generation_metadata["token_usage"]["prompt_tokens"] == 0
    assert guard.last_generation_metadata["ended_with_eos"] is None
