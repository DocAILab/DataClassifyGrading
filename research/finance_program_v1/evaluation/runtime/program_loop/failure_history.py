"""Deduplicated actual failures; one object per source rollout, never a global cache."""

from copy import deepcopy
import hashlib
import json
from .contracts import json_snapshot
from agent.orchestration.validator import resolve_reference


def fingerprint(value) -> str:
    encoded = json.dumps(
        json_snapshot(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class FailureHistory:
    def __init__(self) -> None:
        self._entries: dict[tuple, dict] = {}

    def record(self, tool: str, error: str, input_sha256: str, kind: str) -> None:
        error = str(error)[:600]
        identity = (tool, input_sha256, kind, error)
        entry = self._entries.setdefault(
            identity,
            {
                "tool": tool,
                "error": error,
                "effective_input_sha256": input_sha256,
                "failure_kind": kind,
                "occurrences": 0,
                "authority": "historical execution fact, not new business evidence or proof of current readiness",
            },
        )
        entry["occurrences"] += 1

    def observe_step(self, step: dict, record, before_state: dict) -> None:
        if step["status"] == "success":
            return
        canonical = (step.get("binding") or {}).get("normalized_call") or step["call"]
        try:
            values = {
                key: resolve_reference(value, record, before_state)[1]
                for key, value in canonical.get("arguments", {}).items()
            }
            identity = fingerprint(
                {"tool": canonical["tool"], "resolved_inputs": values}
            )
            identity_kind = "resolved_inputs"
        except (ValueError, TypeError, KeyError):
            identity = fingerprint(canonical)
            identity_kind = "unexecuted_normalized_call"
        kind = step.get("metadata", {}).get("failure_kind", step["status"])
        self.record(step["call"]["tool"], step.get("summary", ""), identity, kind)
        step["control_input_identity"] = {"sha256": identity, "kind": identity_kind}

    def snapshot(self) -> list[dict]:
        return deepcopy(list(self._entries.values()))
