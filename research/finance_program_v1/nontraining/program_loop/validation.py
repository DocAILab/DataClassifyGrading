"""Validate possible dependencies, never pretend a future retrieval succeeded."""
import json
from .contracts import Program, ToolCall
from pool_metadata import FIELDS
from pool_runtime import SEMANTIC_KEYS, ALIGNMENT_KEYS

SELECTION = {"category", "aligned_category", "candidate_category", "category_candidate", "aligned_standard_entry", "grading_level", "grading_level_candidate", "final_evidence_state", "final_result"}
GRADE = {"grading_level", "grading_level_candidate", "final_evidence_state", "final_result"}
EVIDENCE_BUNDLES = {"semantic_query", "semantic_evidence", "current_semantic_state", "current_evidence", "current_evidence_state", "all_available_evidence", "available_evidence", "evidence", "evidence_set"}
FINAL_EVIDENCE_REQUIREMENTS = frozenset({"grading_level", "category", "aligned_standard_entry", "standard_candidates"})


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result: raise ValueError(f"duplicate JSON key {key}")
        result[key] = value
    return result


def canonical_reference(key, available):
    if key == "current_record": return "$record"
    if key == "domain": return "$domain"
    if key in ("field_name", "target_field"): return "$metadata.field_name"
    if key == "candidate_standard_categories" and "standard_candidates" in available: return "$state.standard_candidates"
    return "$state." + key if key in available else None


_reference = canonical_reference


def explicit_value_policy(reference):
    return "canonical_reference_only" if reference and reference.startswith("$state.") else "initial_binding_contract"


def possible_outputs(spec):
    produced = set(spec.produces)
    outputs = set(produced)
    if produced & set(SEMANTIC_KEYS): outputs.update(EVIDENCE_BUNDLES)
    if "field_metadata" in produced:
        outputs.update((*FIELDS, "current_metadata", "available_metadata", "semantic_query", "available_evidence", "all_available_evidence"))
    if "category" in produced: outputs.update(("aligned_category", "candidate_category", "category_candidate"))
    return outputs


def advance_available(spec, available):
    available = set(available)
    produced = set(spec.produces)
    if produced & (set(SEMANTIC_KEYS) | set(ALIGNMENT_KEYS) | {"field_metadata"}): available -= SELECTION
    if "standard_candidates" in produced: available -= SELECTION
    if "category" in produced: available -= GRADE
    available.update(possible_outputs(spec))
    if FINAL_EVIDENCE_REQUIREMENTS <= available: available.add("final_evidence_state")
    return available


def parse_program(raw, catalog, state, max_steps):
    if not isinstance(raw, str): raise ValueError("Planner must return JSON text")
    value = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=lambda v: (_ for _ in ()).throw(ValueError(f"nonfinite JSON {v}")))
    if not isinstance(value, dict) or set(value) != {"program"}: raise ValueError("Expected exactly one JSON Program object")
    steps = value["program"]
    if not isinstance(steps, list) or not steps: raise ValueError("Program must be nonempty")
    if len(steps) > max_steps: raise ValueError("Program length exceeds max_program_steps")
    available = set(state)
    calls = []
    for index, item in enumerate(steps):
        if not isinstance(item, dict) or set(item) != {"tool", "arguments"}: raise ValueError(f"step {index}: expected only tool and arguments")
        name, args = item["tool"], item["arguments"]
        if not isinstance(name, str) or name not in catalog: raise ValueError(f"step {index}: unknown Skill")
        spec = catalog[name]
        if not isinstance(args, dict) or set(args) - set((*spec.required_inputs, *spec.optional_inputs)): raise ValueError(f"step {index}: unknown arguments")
        for key in set(spec.required_inputs) | set(args):
            reference = canonical_reference(key, available)
            if reference is None: raise ValueError(f"step {index}: missing dependency {key}")
            supplied = args.get(key)
            if key in args and ((explicit_value_policy(reference) == "canonical_reference_only" and supplied != reference)
                                or (isinstance(supplied, str) and supplied.startswith("$") and supplied != reference)):
                raise ValueError(f"step {index}: incorrect canonical reference for {key}; expected {reference}")
        if spec.terminal and index != len(steps) - 1: raise ValueError(f"step {index}: terminal Skill must be last")
        available = advance_available(spec, available)
        calls.append(ToolCall(name, args))
    return Program(tuple(calls))
