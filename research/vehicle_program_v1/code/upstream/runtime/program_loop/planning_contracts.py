"""Value-free planning guidance from catalog contracts and shared static rules.

Possible producers and outputs are conditional, never proof of execution success.
Actual-state binding and the executor still validate every supplied input/output.
"""
from .validation import canonical_reference, explicit_value_policy, possible_outputs, FINAL_EVIDENCE_REQUIREMENTS


def planning_contract(spec, catalog):
    known = {key for entry in catalog.values() for key in (*entry.required_inputs, *entry.optional_inputs)}
    known.update(key for entry in catalog.values() for key in possible_outputs(entry))
    known.add("final_evidence_state")
    inputs = {}
    for key in (*spec.required_inputs, *spec.optional_inputs):
        ref = canonical_reference(key, known)
        target = ref.removeprefix("$state.") if ref and ref.startswith("$state.") else None
        inputs[key] = {
            "required": key in spec.required_inputs,
            "canonical_reference": ref,
            "explicit_value_policy": "observed_missing_metadata_type" if spec.name == "ACQUIRE_MISSING_METADATA" and key == "missing_metadata_type" else explicit_value_policy(ref),
            "source": "state" if target else "initial_binding",
            "possible_producers": [other.name for other in catalog.values() if target in possible_outputs(other)] if target and key != "final_evidence_state" else [],
            "requires_all": sorted(FINAL_EVIDENCE_REQUIREMENTS) if key == "final_evidence_state" else [],
        }
        if spec.name == "ACQUIRE_MISSING_METADATA" and key == "missing_metadata_type":
            inputs[key]["choice_rule"] = ("Choose an explicit native type from actual missing_metadata_types, "
                                           "or a type conditionally reported missing by a preceding metadata check/acquisition "
                                           "in this Program. The runtime binder checks actual missing_metadata_types; "
                                           "a future retrieval is not guaranteed to succeed.")
    return {"argument_template": {}, "inputs": inputs, "conditional_outputs": sorted(possible_outputs(spec))}
