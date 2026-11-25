"""Whole native F66 loop with per-instance evidence gaps and explicit abstention."""
from collections import Counter
from copy import deepcopy
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


original = load('gap_original_engine', ROOT.parent / 'finance_program_teacher_evidence_20261020_r1/engine.py')
wide, old = original.wide, original.old
boundary = load('gap_boundary_impl', ROOT / 'boundary.py')


def rollout(item, study, generate, snapshot):
    """Keep W's bounds/step semantics; stop only after an actual locally blocked call."""
    guarded = boundary.Guard(generate)
    snapshot = deepcopy(snapshot)
    local = deepcopy(study)
    local['budget'] = dict(max_programs=6, max_program_steps=12, max_skill_steps=16,
                           max_planner_attempts=6, max_identical_calls=1)
    ctrl = old.c.controller(guarded, local)
    native_step = ctrl.bridge.step

    def step(call, record, state, counts, lineage, failures=None):
        guarded.record, guarded.state = record, state
        before = boundary.evidence_view(record, state)
        n = len(guarded.audits)
        actual, completion = boundary.complete_optional(call, record, state, ctrl.bridge.catalog)
        result = native_step(actual, record, state, counts, lineage, failures)
        after = boundary.evidence_view(record, state)
        result['planner_call'] = call.to_dict()
        result['input_completion'] = completion
        result['evidence_audits'] = deepcopy(guarded.audits[n:])
        result['new_raw_fact_ids'] = [k for k, v in after['facts'].items()
            if k not in before['facts'] or v['value'] != before['facts'][k]['value']]
        result['new_model_interpretation_keys'] = [k for k in after['hypotheses'] if k not in before['hypotheses']]
        return result

    ctrl.bridge.step = step
    try:
        from agent.orchestration.contracts import MetadataRecord
        from program_loop.validation import parse_program
        from program_loop.failure_policy import ComparisonFailureLedger
        record = MetadataRecord.from_mapping(item['initial'])
        state, counts, lineage = snapshot['state'], Counter(snapshot['counts']), snapshot['lineage']
        failures = ComparisonFailureLedger()
        failures.entries = deepcopy(snapshot['comparison_entries'])
        feedback = snapshot['feedback']
        trace = dict(mode='whole', state_id=item['id'], source_id=item['source_id'], initial=deepcopy(item['initial']),
            seed_sha256=old.c.sha(old.c.raw_json(snapshot)), initial_state=deepcopy(state), planner_attempts=[],
            programs=[], new_steps=0, terminated=False, final_result=None, stop_reason='planner_budget_exhausted')
        for attempt_number in range(1, 7):
            if trace['new_steps'] >= 16:
                trace['stop_reason'] = 'skill_budget_exhausted'; break
            if guarded.halted:
                trace['stop_reason'] = 'transport_or_case_budget_halt'; break
            remaining = wide.limits_remaining(attempt_number, trace['new_steps'])
            messages = old.messages_for(ctrl, record, state, remaining, feedback, 'whole')
            attempt = dict(attempt=attempt_number, messages=deepcopy(messages), status='error', raw_response=None)
            trace['planner_attempts'].append(attempt)
            try:
                raw = guarded(messages)
                attempt['raw_response'], attempt['status'] = raw, 'rejected'
                plan = parse_program(raw, ctrl.bridge.catalog, state, remaining['program_steps'])
            except Exception as exc:
                attempt['error_type'] = type(exc).__name__
                attempt['error'] = str(exc) if isinstance(exc, ValueError) else 'backend_error_details_retained_in_call_record'
                feedback = dict(status='planner_rejected', error=attempt['error'], remaining_budget=remaining)
                continue
            attempt['status'] = 'accepted'
            obs = dict(program_index=len(trace['programs']) + 1, attempt=attempt_number, plan=plan.to_dict(), steps=[],
                status='completed', failure_index=None, terminated=False, invalidations=[])
            trace['programs'].append(obs)
            for index, call in enumerate(plan.steps):
                if trace['new_steps'] >= 16:
                    obs.update(status='budget_exhausted', failure_index=index); break
                trace['new_steps'] += 1
                result = ctrl.bridge.step(call, record, state, counts, lineage, failures)
                result['step_index'] = index
                obs['steps'].append(result)
                obs['invalidations'].extend(result['invalidations'])
                if result['status'] != 'success':
                    obs.update(status='failed', failure_index=index); break
                if ctrl.bridge.catalog[call.tool].terminal:
                    trace.update(terminated=True, final_result=deepcopy(result['data']['final_result']), stop_reason='validated_terminal')
                    obs.update(status='terminated', terminated=True); break
            obs['unexecuted_tail'] = deepcopy(obs['plan']['program'][len(obs['steps']):])
            obs['remaining_budget'] = wide.limits_remaining(attempt_number + 1, trace['new_steps'])
            obs['lineage'] = dict(lineage)
            feedback = deepcopy(obs)
            if guarded.stop_reason:
                trace['stop_reason'] = guarded.stop_reason
                break
            if trace['terminated']:
                break
        trace.update(state=deepcopy(state), final_lineage=dict(lineage), planner_calls=len(trace['planner_attempts']),
            evidence_audits=deepcopy(guarded.audits), gap_ledger=deepcopy(guarded.gap_ledger),
            evidence_protocol='acquired-evidence-boundary-v1', gap_protocol='persistent-evidence-gap-v1')
        return trace
    finally:
        ctrl.bridge.step = native_step
