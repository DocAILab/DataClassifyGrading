"""Same frozen execution engine,100 public initial targets, no baseline arm."""
import importlib.util,json,sys
from copy import deepcopy
from pathlib import Path
from common import ROOT,PARENT
from initial_identity import load_inputs
sys.path.append(str(PARENT))
spec=importlib.util.spec_from_file_location('n100_original_native_eval',PARENT/'eval_runtime.py');parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
ContextBackend,CPUReplay,raw_bytes=parent.ContextBackend,parent.CPUReplay,parent.raw_bytes
TechnicalFailure,BudgetExhausted,TargetReached=parent.TechnicalFailure,parent.BudgetExhausted,parent.TargetReached
require=parent.require
def prepare():
 import runtime
 inputs=load_inputs(ROOT/'eval-inputs.json');cases=deepcopy(inputs['cases']);rollout=runtime.build(parent.load_engine(parent.E2))
 for c in cases:
  messages=[]
  def capture(m):messages.append(deepcopy(m));raise TargetReached()
  try:rollout(c['item'],inputs['study'],capture,c['snapshot'])
  except TargetReached:pass
  assert len(messages)==1;c['messages']=messages[0]
 return inputs['study'],cases,rollout
def preflight():
 study,cases,rollout=prepare();assert len(cases)==100
 for c in cases:
  backend=ContextBackend(c,None,None,Path('.'),probe=True)
  try:rollout(c['item'],study,backend,c['snapshot'])
  except TargetReached:pass
  assert backend.target_exact and not backend.records
 return dict(cases=100,exact_first_targets=100,new_model_calls=0)
