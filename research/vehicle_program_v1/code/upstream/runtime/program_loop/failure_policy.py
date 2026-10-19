"""Run-local comparison failure facts and limits; never execute a retry."""
from copy import deepcopy
import hashlib
import json

from .contracts import json_snapshot


COMPARISON = "COMPARE_STANDARD_CANDIDATES"
CORRECTABLE = {"invalid_output", "output_truncated"}
FAILURE_KINDS = CORRECTABLE | {"no_match", "backend_error", "unknown_error"}


def content_sha256(value):
    serialized = json.dumps(json_snapshot(value), sort_keys=True, ensure_ascii=False,
                            separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def raw_sha256(raw):
    if isinstance(raw, str):
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
    if isinstance(raw, bytes):
        return hashlib.sha256(raw).hexdigest()
    return content_sha256(raw)


class ComparisonFailureLedger:
    def __init__(self):
        self.entries = {}

    def before_call(self, inputs):
        """Return identity and actual correction facts, or block exhausted identity."""
        fingerprint = content_sha256(inputs)
        entry = self.entries.get(fingerprint)
        if entry is None:
            return fingerprint, None
        failures = entry["failures"]
        if any(failure["failure_kind"] == "no_match" for failure in failures):
            raise ValueError("comparison blocked: observed no_match for identical effective inputs")
        correctable = [failure for failure in failures if failure["failure_kind"] in CORRECTABLE]
        if len(correctable) >= 2:
            raise ValueError("comparison blocked: explicit correction already failed for identical effective inputs")
        return fingerprint, deepcopy(correctable[-1]) if correctable else None

    def record_failure(self, fingerprint, error, metadata):
        kind = metadata.get("failure_kind", "unknown_error")
        if kind not in FAILURE_KINDS:
            kind = "unknown_error"
        entry = self.entries.setdefault(fingerprint, {"tool": COMPARISON,
            "input_fingerprint": fingerprint, "failure_count": 0, "failures": []})
        entry["failure_count"] += 1
        raw = metadata.get("raw_response")
        entry["failures"].append({"failure_kind": kind, "attempt": entry["failure_count"],
            "request_sha256": content_sha256(metadata["messages"]) if "messages" in metadata else None,
            "raw_sha256": raw_sha256(raw), "raw_response": json_snapshot(raw),
            "validation_error": str(error), "generation": json_snapshot(metadata.get("generation", {}))})

    def snapshot(self):
        return deepcopy(list(self.entries.values()))
