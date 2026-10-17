import importlib
import json
from copy import deepcopy

from test_program_loop import FULL, program, run


def context(state, budget, feedback):
    module = importlib.import_module('program_loop.planner')
    assert hasattr(module, 'recovery_context'), 'factual recovery context is missing'
    return module.recovery_context(state, budget, feedback)


def planner_payloads(script):
    return [json.loads(m[1]['content']) for m in script.messages if 'Global Program Planner' in m[0]['content']]


def test_pure_context_preserves_inputs_and_summarizes_only_execution_facts():
    state = {'category': 'stale-looking', 'final_evidence_state': {}, 'availability_ledger': [
        {'scope': 'record', 'target': 'PRIVATE-ID', 'type': 'field_description', 'status': 'error', 'source': 'PRIVATE-ID'}]}
    budget = {'programs': 1, 'planner_attempts': 2, 'skill_steps': 8, 'program_steps': 10}
    feedback = {'status': 'failed', 'failure_index': 1, 'terminated': False,
        'plan': {'program': [{'tool': t, 'arguments': {'private': 'PRIVATE-ID'}} for t in FULL]},
        'steps': [{'step_index': 0, 'call': {'tool': FULL[0]}, 'status': 'success'},
                  {'step_index': 1, 'call': {'tool': FULL[1]}, 'status': 'error'}],
        'invalidations': ['grading_level', 'final_evidence_state']}
    original = deepcopy((state, budget, feedback))
    result = context(state, budget, feedback)
    assert (state, budget, feedback) == original
    assert result['previous_program']['successful_steps'] == [{'step_index': 0, 'tool': FULL[0]}]
    assert result['previous_program']['failed_steps'] == [{'step_index': 1, 'tool': FULL[1], 'status': 'error'}]
    assert result['previous_program']['unexecuted_tail'] == [{'step_index': i, 'tool': FULL[i]} for i in range(2, 5)]
    assert result['classification_completed'] is False
    assert result['remaining_budget'] == budget and result['invalidations'] == feedback['invalidations']
    assert result['availability_observations'][0]['status'] == 'error'
    assert 'PRIVATE-ID' not in json.dumps(result, allow_nan=False)
    result['remaining_budget']['programs'] = 99
    assert budget['programs'] == 1


def test_rejected_or_error_attempt_has_no_invented_program():
    for status in ('planner_rejected', 'planner_error'):
        result = context({'category': 'cat1', 'grading_level': 'L2'}, {'programs': 1}, {'status': status, 'error': 'bad'})
        assert result['previous_program'] is None and not result['classification_completed']


def test_failure_recovers_actual_state_to_terminal_without_mid_program_planner():
    bad = json.dumps({'option_id': 'C006', 'summary': 'outside'})
    _, trace, script = run([program(*FULL), program(*FULL[2:])],
        overrides={'INFER_CATEGORY_FROM_TOPK': [bad, bad, json.dumps({'option_id': 'C001', 'summary': 'recovered'})]}, max_programs=2)
    payloads = planner_payloads(script)
    assert 'recovery_context' in payloads[1], 'outer planner needs factual recovery context'
    recovery = payloads[1]['recovery_context']
    assert recovery['previous_program']['status'] == 'failed'
    assert recovery['previous_program']['failure_index'] == 2
    assert recovery['previous_program']['unexecuted_tail'] == [{'step_index': 3, 'tool': FULL[3]}, {'step_index': 4, 'tool': FULL[4]}]
    assert recovery['remaining_budget']['programs'] == 1 and not recovery['classification_completed']
    assert payloads[1]['feedback'] == trace.programs[0]
    assert trace.terminated and trace.budget['skill_steps'] == 6
    assert script.kinds.count('planner') == 2
    assert script.kinds[:5] == ['planner', FULL[0], FULL[1], FULL[2], FULL[2]]
    assert all(x not in json.dumps(script.messages) for x in ('PRIVATE-ROW-997', 'SECRET-GOLD', 'SECRET-GRADE'))


def test_one_recovery_step_still_exhausts_program_budget_and_retains_failure():
    bad = json.dumps({'option_id': 'C006', 'summary': 'outside'})
    _, trace, script = run([program(*FULL), program(FULL[2])],
        overrides={'INFER_CATEGORY_FROM_TOPK': [bad, bad, json.dumps({'option_id': 'C001', 'summary': 'recovered'})]}, max_programs=2)
    assert 'recovery_context' in planner_payloads(script)[1]
    assert not trace.terminated and trace.stop_reason == 'program_budget_exhausted'
    assert len(trace.programs) == 2 and trace.programs[0]['status'] == 'failed'
    assert trace.budget['programs'] == 2 and trace.budget['skill_steps'] == 4


def test_real_invalidation_and_source_error_remain_facts_without_readiness_claims():
    _, trace, script = run([program(*FULL[:-1]), program('INTERPRET_DOMAIN'), program(*FULL[1:])])
    assert 'recovery_context' in planner_payloads(script)[2]
    result = planner_payloads(script)[2]['recovery_context']
    assert {'category', 'grading_level', 'final_evidence_state'} <= set(result['invalidations'])
    assert 'category' not in result['observed_state_keys'] and not result['classification_completed']
    assert trace.terminated
    _, missing, script = run([program('GET_FIELD_METADATA'), program('INTERPRET_DOMAIN')], rows=[], max_programs=2)
    result = planner_payloads(script)[1]['recovery_context']
    assert any(x['status'] == 'error' for x in result['availability_observations'])
    assert not any(x['status'] == 'unavailable' for x in result['availability_observations'])
    assert missing.programs[0]['status'] == 'failed'


def test_availability_distinguishes_canonical_unavailable_obtained_and_unqueried():
    result = context({'availability_ledger': [
        {'scope': 'record', 'type': 'field_description', 'status': 'unavailable'},
        {'scope': 'record', 'type': 'field_name', 'status': 'obtained'}]}, {'programs': 1}, None)
    assert result['availability_observations'] == [
        {'scope': 'record', 'type': 'field_description', 'status': 'unavailable'},
        {'scope': 'record', 'type': 'field_name', 'status': 'obtained'}]
    assert 'unqueried' in result['availability_rule']
    assert result['previous_program'] is None
