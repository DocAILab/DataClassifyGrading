"""Complete P14 contracts and snapshots containing only acquired public state."""
from copy import deepcopy
from . import _vendor
from pool_catalog import purpose
from .planning_contracts import planning_contract


def skill_library(catalog, loader):
    return [{"name": spec.name, "purpose": purpose(spec.name), "role": spec.role,
             "required_inputs": list(spec.required_inputs), "optional_inputs": list(spec.optional_inputs),
             "produces": list(spec.produces), "terminal": spec.terminal,
             "contract": loader.load(spec.name).body,
             "planning": planning_contract(spec, catalog)} for spec in catalog.values()]


def pack_library(library):
    """Factor repeated input rules without changing any Skill contract or name."""
    packed = deepcopy(library)
    rules = {}
    for item in packed:
        inputs = item["planning"].pop("inputs")
        for key, details in inputs.items():
            rule = {name: value for name, value in details.items() if name != "required"}
            if key in rules and rules[key] != rule:
                raise ValueError(f"inconsistent planning rule for {key}")
            rules[key] = rule
    return packed, rules


def unpack_library(packed, input_rules):
    """Reconstruct original planning inputs for audit and roundtrip tests."""
    library = deepcopy(packed)
    for item in library:
        required = set(item["required_inputs"])
        item["planning"]["inputs"] = {key: {"required": key in required, **deepcopy(input_rules[key])}
                                          for key in (*item["required_inputs"], *item["optional_inputs"])}
    return library


def public_state(record, state):
    return {"input": {"domain": record.domain, "field_name": record.field_name}, "named_state": deepcopy(state)}
