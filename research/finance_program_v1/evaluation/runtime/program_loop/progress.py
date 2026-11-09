"""Operational novelty in acquired state; it is not a claim of semantic correctness."""
import hashlib
import json

RAW_METADATA = frozenset({"field_name", "field_description", "field_type", "value",
                          "database_name", "database_description", "table_name", "table_description",
                          "context_fields", "context_field_descriptions", "context_field_types",
                          "context_field_values", "table_fields"})
DECISIONS = frozenset({"category", "aligned_category", "grading_level", "final_result"})
CANDIDATES = frozenset({"standard_candidates", "candidate_standard_categories", "candidate_categories"})
CONTROL = frozenset({"availability_ledger", "comparison_failure_ledger", "missing_metadata_types",
                     "missing_metadata_type", "previous_standard_results"})


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def observations(state):
    for key, value in state.items():
        if key.startswith("_") or key in CONTROL:
            continue
        # A first legal intermediate publication is operational progress. Rewording
        # an existing interpretation/fusion is not additional source evidence.
        yield ("first_publication", key)
        if key in RAW_METADATA or key in DECISIONS:
            yield ("observed_value", key, fingerprint(value))
        elif key in CANDIDATES:
            if isinstance(value, list):
                identities = []
                for item in value:
                    identity = next((item[k] for k in ("category", "option_id", "id") if k in item), item) if isinstance(item, dict) else item
                    identities.append(canonical(identity))
                value = sorted(set(identities))
            yield ("candidate_set", key, fingerprint(value))
    for entry in state.get("availability_ledger", []):
        if isinstance(entry, dict) and entry.get("status") in {"obtained", "unavailable", "missing"}:
            value = {k: entry[k] for k in ("scope", "target", "type", "status") if k in entry}
            yield ("availability_observation", fingerprint(value))


class ProgressTracker:
    def __init__(self):
        self._seen = set()
        self.initialized = False

    def seed(self, state):
        if not self.initialized:
            self._seen.update(observations(state))
            self.initialized = True

    def observe(self, state):
        current = set(observations(state))
        novel = current - self._seen
        self._seen.update(current)
        self.initialized = True
        return {"advanced": bool(novel), "novel_observations": len(novel),
                "new_business_facts_supplied_by_control": 0,
                "basis": "first legal publication, novel acquired metadata/availability, candidate set or decision binding; not semantic adjudication"}
