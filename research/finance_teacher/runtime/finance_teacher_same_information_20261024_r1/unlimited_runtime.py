"""Additive native budget-only adapter; no fixed Skill/Program step count cap.

Does not change citations, typed binding, availability, dedup or terminal checks.
Not used by the currently running static-state Teacher collector.
"""
import copy,importlib.util,inspect,json,sys,time,types
from pathlib import Path
ROOT=Path(__file__).resolve().parent
E2=ROOT.parent/'finance_program_student_b80_eval_20261021_r2'
R14=ROOT.parent/'finance_program_termination_repair_20261024_r14'

def remaining_budget(attempt,new_steps):
    return dict(programs=max(0,11-attempt),planner_attempts=max(0,11-attempt),skill_steps=None,program_steps=None)

def unlimited_config():
    return types.SimpleNamespace(max_programs=10,max_program_steps=None,max_skill_steps=None,max_planner_attempts=10,max_identical_calls=1)

def parse_without_step_cap(raw,catalog,state):
    from program_loop.validation import parse_program
    # Count only for the native parser's parameter; no arbitrary length truncation.
    value=json.loads(raw);steps=value.get('program') if isinstance(value,dict) else None
    length=len(steps) if isinstance(steps,list) else 0
    return parse_program(raw,catalog,state,length)

def build(engine,scoped_native_rollout,campaign_stopped):
    source=inspect.getsource(engine.rollout)
    changes=[("local['budget'] = dict(max_programs=6, max_program_steps=12, max_skill_steps=16,\n                           max_planner_attempts=6, max_identical_calls=1)","local['budget'] = unlimited_config()",1),
             ('range(1, 7)','range(1, 11)',1),
             ("trace['new_steps'] >= 16",'campaign_stopped()',2),
             ("trace['stop_reason'] = 'skill_budget_exhausted'","trace['stop_reason'] = 'campaign_budget_exhausted'",1),
             ('wide.limits_remaining(', 'remaining_budget(',2),
             ("parse_program(raw, ctrl.bridge.catalog, state, remaining['program_steps'])",'parse_without_step_cap(raw, ctrl.bridge.catalog, state)',1)]
    for old,new,count in changes:
        if source.count(old)!=count:raise ValueError('native source budget patch shape changed')
        source=source.replace(old,new)
    namespace=dict(scoped_native_rollout.__globals__)
    namespace.update(unlimited_config=unlimited_config,remaining_budget=remaining_budget,parse_without_step_cap=parse_without_step_cap,campaign_stopped=campaign_stopped)
    exec(compile(source,'<native-no-fixed-Skill-or-Program-step-cap>','exec'),namespace)
    return namespace['rollout']

_FIXTURE=None
def _fixture():
    global _FIXTURE
    if _FIXTURE is not None:return _FIXTURE
    sys.path.insert(0,str(E2));sys.path.insert(0,str(R14))
    from paired_eval import load_engine
    engine=load_engine(E2)
    spec=importlib.util.spec_from_file_location('same_information_R14_native_runtime',R14/'runtime.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    scoped=m.build(engine)
    _FIXTURE=(engine,scoped,None,None);return _FIXTURE

def native_test_fixture(expired=False,bad_reference=False,synthetic_bridge=False):
    engine,scoped,study,case=_fixture();rollout=build(engine,scoped,lambda:expired)
    # The synthetic branch tests the loop counter only, never business guards.
    ns=dict(rollout.__globals__);old=ns['old'];old_copy=types.SimpleNamespace(**vars(old));c_copy=types.SimpleNamespace(**vars(old.c));original_create=c_copy.controller
    def create(*args,**kwargs):
        ctrl=original_create(*args,**kwargs)
        if synthetic_bridge:
            def fake_step(call,record,state,counts,lineage,failures=None):return dict(call=call.to_dict(),status='success',data={},binding=None,invalidations=[],lineage=dict(lineage),TestOnly=True)
            ctrl.bridge.step=fake_step
        return ctrl
    c_copy.controller=create;old_copy.c=c_copy;ns['old']=old_copy
    rollout=types.FunctionType(rollout.__code__,ns,rollout.__name__,rollout.__defaults__,rollout.__closure__)
    generated=0;planner=0
    def test_only_backend(messages):
        nonlocal generated,planner
        if messages[0]['content'].startswith('Execute only the selected Skill'):
            generated+=1
            return json.dumps(dict(summary='TestOnly非训练样本',data=dict(field_description_semantics=dict(schema='TestOnly-native-budget-check',status='observed_description',description='证券代码',test_counter=generated))),ensure_ascii=False)
        planner+=1
        if planner>1:return '{}'
        calls=([dict(tool='GET_FIELD_DESCRIPTION',arguments={'field_name':'$wrong'})] if bad_reference else [dict(tool='GET_FIELD_METADATA',arguments={}) for _ in range(41)])
        return json.dumps(dict(program=calls))
    trace=rollout(case['item'],study,test_only_backend,copy.deepcopy(case['snapshot']))
    errors=[a.get('error','') for a in trace['planner_attempts']]
    return dict(new_steps=trace['new_steps'],native_success_steps=0 if synthetic_bridge else sum(s['status']=='success' for p in trace['programs'] for s in p['steps']),stop_reason=trace['stop_reason'],terminated=trace['terminated'],reference_rejected=any('canonical reference' in x for x in errors),new_model_calls=0,mock_backend_returns=generated,program_lengths=[len(p['plan']['program']) for p in trace['programs']],planner_attempts=len(trace['planner_attempts']),first_planner_error=errors[0] if errors else None,first_step_error=next((s.get('summary') for p in trace['programs'] for s in p['steps'] if s['status']!='success'),None),synthetic_bridge_only=synthetic_bridge,native_business_success_claimed=False,TestOnly_not_training_data=True)

def native_static_repeat_test():return native_test_fixture()
def synthetic_loop_only_test():return native_test_fixture(synthetic_bridge=True)

if __name__=='__main__':
    print(json.dumps(dict(unlimited=native_test_fixture(),bad_reference=native_test_fixture(bad_reference=True),expired=native_test_fixture(expired=True)),ensure_ascii=False))
