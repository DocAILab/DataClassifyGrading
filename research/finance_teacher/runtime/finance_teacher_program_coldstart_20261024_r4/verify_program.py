"""Independent raw-provider and native actual-message replay; no new generation."""
import argparse,copy,json,time
from pathlib import Path
from campaign import *
from runtime import native_context,build,CommonActor

class ReplayMismatch(BaseException):pass
class Replay:
 def __init__(self,folder,config,budget_end):
  self.folders=[p for p in sorted(Path(folder).glob('call-*')) if (p/'request.json').exists() and (p/'outcome.json').exists()];self.config=config;self.budget_end=budget_end;self.pos=0;self.halted=False;self.actor=CommonActor();self.last_generation_metadata={};self.last_call_evidence={}
 def stopped(self):return self.halted or self.budget_end and not self.folders
 def __call__(self,messages):
  if self.pos>=len(self.folders):
   if self.budget_end:self.halted=True;raise RuntimeError('campaign_budget_halted')
   raise ReplayMismatch('native replay requested an unrecorded model call')
  folder=self.folders[self.pos];q=read(folder/'request.json');o=read(folder/'outcome.json');actual=self.actor.project(messages);body=build_body(actual['messages'],self.config)
  if q['native_messages']!=messages or q['native_input_sha256']!=message_sha(messages) or q['input_sha256']!=actual['input_sha256'] or q['student_view_sha256']!=actual['input_sha256'] or q['body']!=body or q['body_sha256']!=sha(canon(body)):raise ReplayMismatch('actual native/shared input differs')
  content=o['result'].get('content','')
  if o['provider_raw_content']!=content or o['provider_target_sha256']!=sha(content.encode()):raise ReplayMismatch('actual target repaired')
  self.pos+=1;self.last_generation_metadata=copy.deepcopy(o['metadata']);self.last_call_evidence=copy.deepcopy(o['evidence'])
  if o['backend_exception']:self.halted=True;raise RuntimeError(o['backend_exception'])
  return content
