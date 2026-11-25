"""Project in-memory Agent Loop traces into label-isolated run artifacts."""

from __future__ import annotations

from typing import Any, Mapping


def project_trace(row: Mapping[str, Any]) -> dict[str, Any]:
    """Split one scored row into observation, model-log, trajectory, and evaluation views."""

    source_id = row.get("source_id")
    trace = row.get("trace") or {}
    observations: list[dict[str, Any]] = []
    model_calls: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    for event in trace.get("events", []):
        if not isinstance(event, Mapping):
            continue
        kind = event.get("kind")
        turn = event.get("turn")
        data = event.get("data") or {}
        if not isinstance(data, Mapping):
            continue
        if kind == "policy_action":
            call = data.get("call")
            if isinstance(call, Mapping):
                actions.append({"turn": turn, "tool": call.get("tool"),
                                "arguments": dict(call.get("arguments") or {})})
            model_calls.append({
                "source_id": source_id, "turn": turn, "kind": "policy_action",
                "skill": (call or {}).get("tool") if isinstance(call, Mapping) else None,
                "messages": data.get("messages"), "raw_response": data.get("raw_response"),
                "metadata": data.get("metadata") or {},
            })
        elif kind == "observation":
            observations.append({
                "source_id": source_id, "turn": turn, "skill": data.get("skill"),
                "status": data.get("status"), "summary": data.get("summary"),
                "data": dict(data.get("data") or {}),
            })
            metadata = data.get("metadata") or {}
            if metadata:
                model_calls.append({
                    "source_id": source_id, "turn": turn, "kind": "skill_execution",
                    "skill": data.get("skill"), "metadata": dict(metadata),
                })
    trajectory = {
        "source_id": source_id,
        "input": dict(trace.get("input") or {}),
        "actions": actions,
        "observations": [{key: value for key, value in item.items() if key != "source_id"}
                         for item in observations],
        "final_result": trace.get("final_result"),
        "stop_reason": trace.get("stop_reason"),
    }
    evaluation = {
        "source_id": source_id,
        "ground_truth": row.get("ground_truth"),
        "ground_truth_level": row.get("ground_truth_level"),
        "predicted_category": row.get("predicted_category"),
        "predicted_level": row.get("predicted_level"),
        "category_correct": row.get("category_correct"),
        "level_correct": row.get("level_correct"),
        "joint_correct": row.get("joint_correct"),
    }
    return {"observations": observations, "model_calls": model_calls,
            "trajectory": trajectory, "evaluation": evaluation}
