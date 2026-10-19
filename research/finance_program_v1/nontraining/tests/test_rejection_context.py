import json
import pytest
from test_program_loop import FULL, program, run

NINE = ('GET_FIELD_METADATA','INTERPRET_FIELD_DESCRIPTION','INTERPRET_FIELD_NAME',
        'DISAMBIGUATE_FIELD_SEMANTICS','SELECT_CATEGORY_TOPK','COMPARE_STANDARD_CANDIDATES',
        'INFER_CATEGORY_FROM_TOPK','INFER_GRADING_LEVEL','FINALIZE_CLASSIFICATION')

def payload(trace, index):
    return json.loads(trace.planner_attempts[index]['messages'][1]['content'])

def test_nine_step_rejection_is_factual_and_next_whole_program_can_terminate():
    raw = program(*NINE)
    _, trace, script = run([raw, program(*FULL)], max_program_steps=8,
                          max_programs=3, max_skill_steps=12, max_planner_attempts=4)
    first, second = payload(trace, 0), payload(trace, 1)
    assert trace.planner_attempts[0]['status'] == 'rejected'
    assert trace.planner_attempts[0]['raw_response'] == raw
    assert second['feedback']['rejected_program'] == {
        'executed': False,'trusted': False,'step_count': 9,'max_program_steps': 8,
        'tool_sequence': list(NINE)}
    assert second['feedback']['error'] == 'Program length exceeds max_program_steps'
    assert second['feedback']['remaining_budget'] == {
        'programs':3,'planner_attempts':3,'skill_steps':12,'program_steps':8}
    assert second['current'] == first['current']
    assert second['recovery_context']['previous_program'] is None
    assert not second['recovery_context']['classification_completed']
    assert trace.terminated and trace.stop_reason == 'validated_terminal'
    assert trace.budget == {'programs':1,'planner_attempts':2,'skill_steps':5}
    assert len(trace.programs) == 1
    assert [s['call']['tool'] for s in trace.programs[0]['steps']] == list(FULL)
    assert script.kinds[:2] == ['planner','planner']

def test_rejected_parameters_are_not_rebroadcast_or_published():
    raw = program({'tool':'INTERPRET_FIELD_NAME','arguments':{'field_name':'$state.PRIVATE-FAKE'}},
                  *NINE[1:])
    _, trace, _ = run([raw, program(*FULL)], max_program_steps=8)
    second = payload(trace, 1)
    assert second['feedback']['rejected_program']['tool_sequence'] == ['INTERPRET_FIELD_NAME',*NINE[1:]]
    assert 'PRIVATE-FAKE' not in json.dumps(second)
    assert 'PRIVATE-FAKE' in trace.planner_attempts[0]['raw_response']
    assert 'PRIVATE-FAKE' not in json.dumps(trace.state)
    assert trace.terminated

@pytest.mark.parametrize('raw', [
    'not JSON','{}','[]','{"program":[]}',
    '{"program":[],"program":[]}',
    '{"program":[{"tool":"INTERPRET_DOMAIN","arguments":{"x":NaN}}]}',
    '{"program":[{"tool":"INTERPRET_DOMAIN","arguments":{"x":Infinity}}]}',
    '{"program":[{"tool":"INTERPRET_DOMAIN","arguments":{"x":1e999}}]}',
    '{"program":[{"tool":"INTERPRET_DOMAIN","arguments":{"x":-1e999}}]}',
    '{"program":[{"tool":"INTERPRET_DOMAIN","arguments":{"x":[{"v":1e999}]}}]}',
    '{"program":[{"tool":"INTERPRET_DOMAIN","tool":"INTERPRET_FIELD_NAME","arguments":{}}]}',
    program('PRIVATE-UNKNOWN'),
    '{"program":[{"tool":"INTERPRET_DOMAIN","arguments":[]}]}',
    '{"program":[{"tool":"INTERPRET_DOMAIN","arguments":{},"extra":1}]}',
    b'not text',None])
def test_unreadable_or_unrecognized_plan_has_no_invented_summary(raw):
    _, trace, _ = run([raw, program(*FULL)], max_planner_attempts=2)
    second = payload(trace, 1)
    assert second['feedback']['status'] == 'planner_rejected'
    assert 'rejected_program' not in second['feedback']
    assert second['recovery_context']['previous_program'] is None
    assert trace.terminated and trace.budget['programs'] == 1

def test_finite_exponent_is_still_summarized_as_unexecuted_intent():
    raw = '{"program":[{"tool":"INTERPRET_DOMAIN","arguments":{"x":1e3}}]}'
    _, trace, _ = run([raw, program(*FULL)], max_planner_attempts=2, max_program_steps=8)
    second = payload(trace, 1)
    assert second['feedback']['rejected_program'] == {
        'executed':False,'trusted':False,'step_count':1,'max_program_steps':8,
        'tool_sequence':['INTERPRET_DOMAIN']}
    assert second['feedback']['error'] == 'step 0: unknown arguments'
    assert trace.terminated and trace.budget['programs'] == 1

def test_backend_error_does_not_reuse_last_rejected_program():
    _, trace, _ = run([program(*NINE), RuntimeError('temporary backend failure'), program(*FULL)],
                     max_program_steps=8,max_planner_attempts=3)
    assert 'rejected_program' in payload(trace, 1)['feedback']
    assert payload(trace, 2)['feedback'] == {'status':'planner_error',
        'error':'temporary backend failure','remaining_budget':{
            'programs':4,'planner_attempts':1,'skill_steps':20,'program_steps':8}}
    assert payload(trace, 2)['recovery_context']['previous_program'] is None
    assert trace.terminated

def test_repeated_invalid_plan_never_executes_or_manufactures_termination():
    _, trace, _ = run([program(*NINE)]*4,max_program_steps=8,
                     max_programs=3,max_skill_steps=12,max_planner_attempts=4)
    assert trace.budget == {'programs':0,'planner_attempts':4,'skill_steps':0}
    assert trace.programs == [] and not trace.terminated and trace.final_result is None
    assert trace.stop_reason == 'planner_budget_exhausted'
    assert all(payload(trace, i)['current'] == payload(trace, 0)['current'] for i in range(4))
