"""Controller-owned uncertain decisions, checked by the unchanged native finalizer.

This is a new termination policy, not a claim of newly sufficient evidence.
No model output, citation, metadata, gap ledger or registry entry is rewritten.
"""
from collections import Counter
from copy import deepcopy

POLICY = 'termination-first-v1'
RECOVERABLE = frozenset({
    'unresolved_no_new_business_evidence',
    'planner_budget_exhausted',
    'skill_budget_exhausted',
})
MAX_SKILL_STEPS = 32


def _abstain(trace, annotation, reason, event=None):
    trace.update(terminated=True, final_result=None, stop_reason='abstained_terminal',
                 termination=dict(annotation, outcome='abstained', reason=reason))
    trace['terminal_event'] = event or dict(tool='ABSTAIN_CLASSIFICATION', status='success', model_calls=0)
    return trace


def finish(trace, controller, record):
    """Finish a recoverable native rollout transactionally; preserve strict success.

    The fallback decision chooses the first valid *current ranked* candidate,
    binds its level from the frozen registry, and calls the real native executor.
    It never publishes the selection under an INFER_* provenance or fabricates a
    sufficient support object. Only a successful executor result and terminal
    guard allow a prediction. A missing candidate or rejected finalizer abstains.
    """
    if trace.get('terminated') or trace.get('stop_reason') not in RECOVERABLE:
        return trace
    if any(a.get('status') == 'generation_error' and a.get('error_kind') == 'backend_error'
           for a in trace.get('evidence_audits', [])):
        return trace
    trace = deepcopy(trace)
    annotation = dict(policy=POLICY, confidence='low', requires_review=True,
                      evidence_sufficient=False,
                      original_stop_reason=trace['stop_reason'],
                      selection_rule='first_valid_current_ranked_candidate',
                      new_model_calls=0)
    if trace['new_steps'] >= MAX_SKILL_STEPS:
        return _abstain(trace, annotation, 'no_reserved_terminal_skill_step')
    state = trace['state']
    registry = controller.bridge.guard.registry_entries
    candidates = state.get('standard_candidates')
    if not isinstance(candidates, list) or not candidates:
        return _abstain(trace, annotation, 'no_current_ranked_candidates')
    eligible = [c.get('category') for c in candidates if isinstance(c, dict)
                and isinstance(c.get('category'), str) and c['category'] in registry]
    if not eligible:
        return _abstain(trace, annotation, 'no_registered_current_candidate')
    selected = eligible[0]
    entry = deepcopy(registry[selected])
    level = entry.get('grading_level')
    if not isinstance(level, str) or level not in controller.bridge.guard.approved_levels:
        return _abstain(trace, annotation, 'invalid_frozen_registry_level')
    working = deepcopy(state)
    aligned = dict(entry, category=selected)
    decision = dict(category=selected, grading_level=level, aligned_standard_entry=aligned)
    working.update(decision)
    working.setdefault('_provenance', {}).update({key:POLICY for key in decision})
    working['final_evidence_state'] = controller.bridge.guard._final_evidence_state(record, working)
    working['_provenance']['final_evidence_state'] = POLICY
    from agent.orchestration.contracts import ToolCall
    call = ToolCall('FINALIZE_CLASSIFICATION', {'final_evidence_state':'$state.final_evidence_state'})
    lineage = deepcopy(trace['final_lineage'])
    lineage['selection'] = lineage.get('selection', 0) + 1
    observation = controller.bridge.step(call, record, working, Counter(), lineage)
    event = dict(tool='FINALIZE_BEST_EFFORT', policy=POLICY, selection=selected,
                 grade_source='frozen_registry', observation=deepcopy(observation), model_calls=0)
    trace['new_steps'] += 1
    result = observation.get('data', {}).get('final_result')
    expected = dict(category=selected, grading_level=level)
    if (observation.get('status') != 'success' or result != expected
            or not controller.bridge.guard._valid_terminal(result, working)):
        return _abstain(trace, annotation, 'native_finalizer_rejected', event)
    trace.update(terminated=True, final_result=deepcopy(result), stop_reason='best_effort_terminal',
                 state=working, final_lineage=lineage, terminal_event=event,
                 termination=dict(annotation, outcome='best_effort_prediction',
                                  native_finalizer_executed=True, native_guard_valid=True))
    return trace
