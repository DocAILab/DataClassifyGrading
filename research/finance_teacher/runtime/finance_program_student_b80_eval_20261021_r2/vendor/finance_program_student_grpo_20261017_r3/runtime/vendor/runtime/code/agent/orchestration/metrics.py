"""Auditable metrics for local Program execution traces."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

from .runtime import ExecutionTrace, Observation


def _value(item: Observation | dict[str, Any], key: str, default: Any = None) -> Any:
    if isinstance(item, Observation):
        return getattr(item, key, default)
    return item.get(key, default)


def summarize_trace(trace: ExecutionTrace) -> dict[str, Any]:
    names = [_value(item, "skill", "") for item in trace.observations]
    statuses = [_value(item, "status", "error") for item in trace.observations]
    counts = Counter(names)
    return {
        "tool_calls": len(names),
        "successful_tool_calls": sum(status == "success" for status in statuses),
        "failed_tool_calls": sum(status != "success" for status in statuses),
        "tool_call_success_rate": (sum(status == "success" for status in statuses) / len(statuses)) if statuses else 1.0,
        "program_length": len(names),
        "duplicate_calls": sum(max(count - 1, 0) for count in counts.values()),
        "terminated": bool(trace.terminated),
        "final_result_valid": isinstance(trace.final_result, dict) and bool(trace.final_result),
    }


def summarize_traces(traces: Iterable[ExecutionTrace]) -> dict[str, Any]:
    rows = [summarize_trace(trace) for trace in traces]
    count = len(rows)
    calls = sum(row["tool_calls"] for row in rows)
    successful = sum(row["successful_tool_calls"] for row in rows)
    return {
        "traces": count,
        "termination_rate": (sum(row["terminated"] for row in rows) / count) if count else 0.0,
        "average_program_length": (sum(row["program_length"] for row in rows) / count) if count else 0.0,
        "tool_call_success_rate": (successful / calls) if calls else 1.0,
        "average_duplicate_calls": (sum(row["duplicate_calls"] for row in rows) / count) if count else 0.0,
    }


__all__ = ["summarize_trace", "summarize_traces"]

