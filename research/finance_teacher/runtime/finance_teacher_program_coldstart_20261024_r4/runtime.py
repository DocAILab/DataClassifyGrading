"""Budget-driven native rollout; unchanged typed/evidence/terminal/repeat guards."""
import copy,inspect,json,sys,types
from pathlib import Path
from campaign import ROOT,ANCHOR,R14,load,attempts,remaining,audit_shared,message_sha,format_messages
_CONTEXT=None
def native_context():
 global _CONTEXT
 if _CONTEXT is None:
  old=load('coldstart_native_context_factory',ANCHOR/'unlimited_runtime.py');_CONTEXT=old._fixture()
 return _CONTEXT
def parse_without_step_cap(raw,catalog,state):
 from program_loop.validation import parse_program
 value=json.loads(raw);steps=value.get('program') if isinstance(value,dict) else None
 return parse_program(raw,catalog,state,len(steps) if isinstance(steps,list) else 0)
def unlimited_config():return types.SimpleNamespace(max_programs=None,max_program_steps=None,max_skill_steps=None,max_planner_attempts=None,max_identical_calls=1)
def build(engine,scoped_native,campaign_stopped):
 source=inspect.getsource(engine.rollout)
 changes=[("local['budget'] = dict(max_programs=6, max_program_steps=12, max_skill_steps=16,\n                           max_planner_attempts=6, max_identical_calls=1)","local['budget'] = unlimited_config()",1),('range(1, 7)','attempts(campaign_stopped)',1),("trace['new_steps'] >= 16",'campaign_stopped()',2),("trace['stop_reason'] = 'skill_budget_exhausted'","trace['stop_reason'] = 'campaign_budget_exhausted'",1),("stop_reason='planner_budget_exhausted'","stop_reason='campaign_budget_exhausted'",1),('wide.limits_remaining(','campaign_remaining(',2),("parse_program(raw, ctrl.bridge.catalog, state, remaining['program_steps'])",'parse_without_step_cap(raw, ctrl.bridge.catalog, state)',1)]
 for old,new,count in changes:
  if source.count(old)!=count:raise ValueError('native source shape differs; no broad rewrite')
  source=source.replace(old,new)
 # The original scoped controller carries metadata isolation and typed binder.
 ns=dict(scoped_native.__globals__);ns.update(attempts=attempts,campaign_remaining=remaining,campaign_stopped=campaign_stopped,unlimited_config=unlimited_config,parse_without_step_cap=parse_without_step_cap)
 exec(compile(source,'<native-budget-only-no-static-attempt-or-step-cap>','exec'),ns)
 return ns['rollout']
class CommonActor:
 def __init__(self):
  native_context();interface=load('coldstart_R14_current_interface',R14/'interface.py');self.sessions={actor:interface.RecoverySession() for actor in ('student','teacher')}
  from alias_contract import project_messages
  self.scope_projection=project_messages
 def project(self,messages):
  views={}
  for actor,session in self.sessions.items():
   actual=session.rewrite(copy.deepcopy(messages));head=actual[0]['content'].splitlines()[0]
   actual=self.scope_projection(actual,actor)
   views[actor]=format_messages(actual,actor)
  audit_shared(views['student'],views['teacher'])
  return dict(messages=views['teacher'],input_sha256=message_sha(views['teacher']),same_information_exact=True,native_input_sha256=message_sha(messages))
def probe(kind):
 engine,scoped,study,case=native_context();n=[0]
 def stopped():return kind=='expired' or n[0]>=(12 if kind=='attempts12' else 1 if kind=='badref' else 2)
 def backend(messages):
  n[0]+=1
  if kind=='attempts12':return '{}'
  if kind=='badref':return json.dumps(dict(program=[dict(tool='GET_FIELD_DESCRIPTION',arguments=dict(field_name='$wrong'))]))
  if n[0]==1:return json.dumps(dict(program=[dict(tool='GET_FIELD_METADATA',arguments={}) for _ in range(41)]))
  return '{}'
 return build(engine,scoped,stopped)(case['item'],study,backend,copy.deepcopy(case['snapshot']))
