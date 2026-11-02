"""Generate one whole Program per outer attempt."""
import json
from .catalog_context import public_state, pack_library
from .prompt_encoding import pack_prompt
from .recovery_context import recovery_context
from .validation import canonical_reference
from pool_metadata import present

SYSTEM = """You are the Global Program Planner for a database-field classification task.
Plan the complete remaining classification task in one ordered Program, not just the next Skill.
Choose the Skills and their order yourself from the complete P14 library.
Plan all dependencies through a successful terminal; selecting candidates alone does not finish classification.
You need not see a future output now: use its documented canonical reference after its producer; execution will bind the actual verified output.
Conditional outputs may be missing or execution may fail; the Program then stops and its factual observation returns to you at the outer boundary.
The parser accepts partial Programs, but that is a permissive syntax boundary, not the planning objective.
Each Program must fit remaining_budget.program_steps and remaining_budget.skill_steps; these are hard upper bounds, not suggested plan lengths.
Use a sufficient dependency path for the remaining task; complete classification does not require using every Skill or every possible evidence source.
Before returning JSON, count the planned steps and omit nonessential operations if it exceeds either bound, without omitting required dependencies or inventing evidence.
Each possible_producers list contains alternative ways to supply that input, not a sequence that must all be executed; choose a sufficient dependency path and still verify its conditional outputs at execution.
all_available_evidence means the verified evidence actually accumulated by the chosen Skills, not every possible evidence source; only a nonempty requires_all list requires every named prerequisite.
Return exactly one JSON object with only a program key whose value is an ordered array of steps. Each step has only tool (the chosen Skill name) and arguments (an object).
No Skill name is mandatory and the controller inserts no steps.
Only domain and field_name are initially visible. A planned metadata retrieval is not evidence.
Prefer arguments:{} so the existing execution-time binder supplies available required inputs.
To use an available optional input, supply its documented canonical reference explicitly; omitted optional inputs are not automatically inserted.
If an optional input is unavailable, omit it; do not supply null placeholders or invent a value. Explicit supplied parameters remain checked.
Use each Skill's required_inputs and optional_inputs with shared_input_rules to distinguish inputs and obtain each canonical_reference.
Follow shared_input_rules explicit_value_policy: canonical_reference_only requires the exact documented reference for every explicit dynamic input. Never handwrite dynamic state values, including objects, lists, null, numbers, or ordinary strings; omit required inputs to use the runtime binder. observed_missing_metadata_type is the sole native explicit-choice exception and the runtime binder verifies it against actual missing_metadata_types.
For state dependencies, choose an appropriate possible_producers Skill before the consumer, unless the input is already in actual state.
Possible producers and conditional_outputs describe static possibilities, not successful execution evidence. Metadata and other outputs are conditional and validated at execution.
Initial bindings are $domain, $metadata.field_name for field_name/target_field, and internal $record for current_record; $record discloses no private identity.
semantic_query is a query input; semantic_evidence is semantic evidence. They have distinct canonical paths and contracts. field_description_semantics does not supply field_semantics.
final_evidence_state requires every prerequisite in shared_input_rules.final_evidence_state.requires_all; no single possible producer guarantees it.
Future references are allowed only after a producing Skill. Explicit references must use the documented canonical path; values are checked against actual state at execution.
Finish only by a successful terminal Skill consuming verified final_evidence_state. No steps after terminal.
Each failure ends the current Program. Use its observation to plan within the remaining global budget.
Recovery context reports execution facts only. Unexecuted tail is historical intent, not required steps. State presence and historical success do not establish current readiness; invalidated outputs must be reacquired or recomputed through your chosen Skills.
Consider the remaining complete task and actual budget when choosing the next whole Program, especially the final available Program: one successful recovery step may still leave classification unfinished. Legal partial Programs remain allowed; no automatic completion or bonus budget is provided.
Never invent metadata, candidates, categories, frozen grades, or ground truth.
For compact planning messages, shared_input_rules applies to every Skill input of the same name; required membership is in each Skill's required_inputs. A {'prompt_ref':'vN'} in current, feedback, or recovery_context means the exact observation in shared_values[vN]; resolve such references recursively. Prompt references are never executable arguments."""


def planner_messages(record, state, library, budget, feedback, *, skill_pool="P14"):
    compact_library, input_rules = pack_library(library)
    payload = {"task": "Plan the complete remaining classification task using verified evidence and frozen Registry grading, through legal terminal execution.",
               "current": public_state(record, state), "skill_library": compact_library,
               "shared_input_rules": input_rules,
               "remaining_budget": budget, "feedback": feedback,
               "recovery_context": recovery_context(state, budget, feedback),
               "availability_rule": "unknown permits acquisition; canonical source observations establish obtained/unavailable; errors remain retryable; omitted batch attributes remain unknown"}
    if skill_pool == "F66" and any(
            failure.get("failure_kind") == "no_match"
            for entry in state.get("comparison_failure_ledger", [])
            for failure in entry.get("failures", [])):
        payload["recovery_context"]["comparison_recovery_rule"] = (
            "An observed no_match blocks another comparison with identical effective inputs. "
            "Repeating the same failed comparison cannot advance the task and still consumes the remaining budget. "
            "no_match does not invalidate the actual retrieved candidates or create a successful alignment; "
            "it records that this comparison found no supported match. A comparison success is not a universal "
            "prerequisite for every classification Skill: inspect each alternative Skill's actual required inputs "
            "and current verified state. Plan the complete remaining task autonomously within the existing budget, "
            "using genuinely new evidence or changed candidates where available, or another contract-valid path "
            "supported by current observations. Do not reuse a blocked comparison, fabricate alignment, or assume "
            "that an unexecuted tail must be followed. Only a verified terminal completes classification.")
    system = SYSTEM if skill_pool == "P14" else SYSTEM.replace("complete P14 library", "complete F66 library")
    if skill_pool == "F66":
        payload["contract_dependency_facts"] = {
            "category_and_alignment_producers": [
                {"name": s["name"], "required_inputs": s["required_inputs"], "produces": s["produces"]}
                for s in library if {"category", "aligned_standard_entry"} <= set(s["produces"])],
            "comparison_dependency_rule": (
                "Comparison is optional for a Skill whose required inputs do not require a comparison output. "
                "The listed category producers themselves publish the selected category and its frozen aligned entry "
                "after a validated choice from actual retrieved candidates. This is a literal contract subset, "
                "not a selected Program or permission to skip any required input.")}
        current_facts = []
        for spec in payload["contract_dependency_facts"]["category_and_alignment_producers"]:
            observed = []
            for key in spec["required_inputs"]:
                reference = canonical_reference(key, state)
                if (reference and reference.startswith("$state.") and
                        present(state.get(reference.removeprefix("$state.")))):
                    observed.append(key)
            current_facts.append({"name": spec["name"], "observed_required_inputs": observed,
                "unobserved_required_inputs": [key for key in spec["required_inputs"] if key not in observed]})
        payload["contract_dependency_facts"]["current_category_input_facts"] = current_facts
        system += (
            "\nF66 dependency reminder: consult contract_dependency_facts and the full contracts. "
            "Do not add a successful comparison as an undocumented prerequisite for category inference. "
            "Choose the dependency path yourself. For runtime state inputs prefer arguments:{}; "
            "the binder supplies actually available required inputs. Never paste observed metadata or semantic "
            "text into dynamic arguments. A missing semantic dependency needs its actual producer before use; "
            "a raw table description is not already table_semantics. Observe all original budgets and terminal guards.")
        system += (
            "\nF66 completion objective: prioritize a complete contract-valid classification path whose required "
            "inputs are already observed. Input presence is factual context, not a guarantee of semantic validity; "
            "execution still verifies all contracts. Optional diagnostics must not become undocumented gating "
            "steps. A no_match observation does not require repeatedly searching until an optional comparison "
            "agrees. Choose the remaining whole Program yourself from actual evidence and dependencies.")
        if feedback and feedback.get("status") == "planner_rejected":
            payload["request_mode"] = "revise_rejected_program"
            payload["revision_request"] = {
                "executed": False,
                "validation_error": feedback.get("error"),
                "allowed_skill_names": [spec["name"] for spec in library],
                "argument_rule": (
                    "Prefer arguments:{} for required runtime-bound inputs. For explicit state inputs copy the "
                    "exact shared_input_rules canonical_reference; do not substitute a different alias or observed "
                    "text. Native observed_missing_metadata_type choices remain subject to their actual contract.")}
            payload["task"] = (
                "Revise the previous unexecuted Program to correct the actual validation error in feedback. "
                "Return one complete remaining Program, using only exact allowed Skill names and canonical "
                "arguments from the contracts. The rejected Program produced no observations. Use current "
                "verified state and the unchanged remaining budget; do not repeat the rejected error or invent "
                "a new Skill or dependency. Choose the sequence yourself and finish through the validated terminal.")
    encoded = pack_prompt(payload)
    if skill_pool == "F66":
        tail = ("current", "feedback", "recovery_context", "remaining_budget", "task")
        encoded = {**{key: value for key, value in encoded.items() if key not in tail},
                   **{key: encoded[key] for key in tail}}
    return [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(encoded, ensure_ascii=False, allow_nan=False, separators=(",", ":"))}]
