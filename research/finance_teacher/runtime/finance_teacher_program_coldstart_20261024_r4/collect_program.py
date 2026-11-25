"""Whole native Teacher episodes; serial real provider calls and immutable raw records."""
import argparse,json,os,re,sys,time,multiprocessing
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from campaign import *
OWN_RUNTIME=load('fullteacher_explicit_own_native_runtime',ROOT/'runtime.py')
native_context,build,CommonActor=OWN_RUNTIME.native_context,OWN_RUNTIME.build,OWN_RUNTIME.CommonActor
from collect import transport_critical_ok

class Backend:
 def __init__(self,folder,config,budget,capture,key,verify_own):
  self.folder=Path(folder);self.config=config;self.budget=budget;self.capture=capture;self.key=key;self.verify_own=verify_own;self.actor=CommonActor();self.count=0;self.bad_format_streak=0;self.last_generation_metadata={};self.last_call_evidence={}
 @property
 def halted(self):return self.budget.halted or time.time()>=self.budget.deadline
 def __call__(self,native_messages):
  if self.halted:raise RuntimeError('campaign_budget_halted')
  try:self.verify_own();actual=self.actor.project(native_messages)
  except Exception as exc:
   self.budget.halted=True;put(self.folder/'identity-halt.json',dict(error_type=type(exc).__name__,native_input_sha256=message_sha(native_messages),new_HTTP=0));raise RuntimeError('shared_actor_or_frozen_identity_halt')
  messages=actual['messages'];body=build_body(messages,self.config);raw=canon(body);self.count+=1;folder=self.folder/f'call-{self.count:04d}';folder.mkdir(parents=True,exist_ok=False)
  stage=re.match(r'Execute only the selected Skill ([A-Z_]+)\.',messages[0]['content']);role='Skill' if stage else 'Planner';stage=stage.group(1) if stage else 'Planner'
  try:reserve=self.budget.admit(len(raw),8192)
  except ValueError:self.budget.halted=True;put(folder/'not-called.json',dict(reason='finite_HTTP_cost_or_deadline',new_HTTP=0));raise RuntimeError('campaign_budget_halted')
  request=dict(role=role,stage=stage,native_messages=native_messages,native_input_sha256=actual['native_input_sha256'],body=body,body_sha256=sha(raw),input_sha256=actual['input_sha256'],student_view_sha256=actual['input_sha256'],same_information_exact=True,endpoint=self.config['endpoint'],reservation_CNY=reserve,global_new_claim=self.budget.claims,time_unix=time.time(),credential_recorded=False);put(folder/'request.json',request)
  remaining=min(180,self.budget.deadline-time.time())
  if remaining<=0:self.budget.halted=True;self.budget.pending=0.;put(folder/'not-called.json',dict(reason='deadline_before_HTTP',new_HTTP=0));raise RuntimeError('campaign_budget_halted')
  put(folder/'invoked.json',dict(time_unix=time.time(),no_HTTP_retry=True))
  try:r=self.capture(folder/'capture',self.config['endpoint']+'/chat/completions',raw,self.key,self.config['model'],remaining)
  except BaseException as exc:r=dict(content='',usage=None,complete=False,error_type=type(exc).__name__,fatal_capture=True)
  self.budget.debit(r.get('usage'));critical=transport_critical_ok(r,self.config['model']);content=r.get('content','')
  if not critical or not isinstance(content,str):self.budget.halted=True
  usage=r.get('usage') or {};self.last_generation_metadata=dict(input_tokens=usage.get('prompt_tokens',0),output_tokens=usage.get('completion_tokens',0),error=None if r.get('complete') else 'provider_output_rejected',truncation_at_budget=r.get('finish_reason')=='length')
  self.last_call_evidence=dict(model=self.config['model'],role=role,input_sha256=actual['input_sha256'],provider_target_sha256=sha(content.encode()) if isinstance(content,str) else None)
  backend_exception='provider_transport_or_identity_halt' if not critical or not isinstance(content,str) else None
  self.bad_format_streak=self.bad_format_streak+1 if not raw_json_object(content) else 0
  if self.bad_format_streak>=self.config.get('consecutive_format_failure_circuit_breaker',3):self.budget.halted=True;backend_exception='repeated_raw_JSON_contract_violation'
  put(folder/'outcome.json',dict(provider_raw_content=content,provider_target_sha256=self.last_call_evidence['provider_target_sha256'],result=r,metadata=self.last_generation_metadata,evidence=self.last_call_evidence,backend_exception=backend_exception,budget_after=self.budget.snapshot(),time_unix=time.time(),credential_recorded=False));print(json.dumps(dict(call=str(folder.relative_to(ROOT)) if folder.is_relative_to(ROOT) else 'CPU_test_fixture',role=role,stage=stage,claimed=self.budget.claims,spent_CNY=self.budget.spent,halted=self.budget.halted)),flush=True)
  if backend_exception:raise RuntimeError(backend_exception)
  return content
