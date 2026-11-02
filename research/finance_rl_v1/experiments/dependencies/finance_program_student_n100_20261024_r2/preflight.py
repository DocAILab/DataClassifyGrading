import json,time
from common import ROOT,PARENT,PREVIOUS
from eval_runtime import prepare,preflight as initial_preflight
import interface
def run():
 native=initial_preflight();study,cases,rollout=prepare()
 from program_loop.validation import parse_program
 import runtime
 from eval_runtime import parent
 controller=runtime.build(parent.load_engine(parent.E2)).__globals__['old'].c.controller(lambda m:'{}',study)
 assert len(parse_program(json.dumps(dict(program=[dict(tool='GET_FIELD_METADATA',arguments={}) for _ in range(13)])),controller.bridge.catalog,{},16).steps)==13
 assert len(rollout(cases[0]['item'],study,lambda m:'{}',cases[0]['snapshot'])['planner_attempts'])==10
 from token_audit import tokenizer
 tok=tokenizer();tokens=[]
 for c in cases:
  m=interface.RecoverySession().rewrite(c['messages']);text=tok.apply_chat_template(m,tokenize=False,add_generation_prompt=True,enable_thinking=False)
  n=len(tok(text,add_special_tokens=False)['input_ids']);assert n+512<=49152;tokens.append(dict(source_id=c['source_id'],tokens=n))
 reuse=json.loads((ROOT/'reuse-lock.json').read_bytes());by={c['source_id']:c for c in cases}
 for r in reuse['rows']:
  p=PREVIOUS/'runs'/r['slot_id']/'call-001/returned.json';old=json.loads(p.read_bytes())['messages'];assert old==by[r['source_id']]['messages']
 from backend import EvaluationBackend
 admission=json.loads((PARENT/'admission.json').read_bytes());admission.update(evaluation_only=True,gpu_deadline_unix=time.time()+3600)
 b=EvaluationBackend(admission);assert b.runtime is None and len(b.expected_tensors)==80
 return dict(native=native,native13steps=True,native10planner=True,initial_tokens=tokens,reused_initial_exact=8,expected_fp32_tensors=80,new_model_calls=0,gpu_loaded=False)
