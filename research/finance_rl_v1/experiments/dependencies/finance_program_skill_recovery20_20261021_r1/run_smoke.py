"""One fixed 20-case student evaluation, same B80 model, no teacher or updates."""
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parent
E2=ROOT.parent/'finance_program_student_b80_eval_20261021_r2'
sys.path.insert(0,str(E2))
from backend import backend_class
from paired_eval import load_inputs,load_engine,ContextBackend,CPUReplay,TargetReached
from prepare_runtime import exclusive_json,raw_bytes,require
from training_contract import check_deadline,file_sha256
from interface import rewrite
from runtime import build,limits_remaining
ModelBackend=backend_class(E2)

def prepare():
    inputs,_=load_inputs(E2);original=load_engine(E2);rollout=build(original)
    cases=deepcopy(inputs['cases'][:20])
    selection=json.loads((ROOT/'selection.json').read_bytes())
    require([c['source_id'] for c in cases]==selection['source_ids'],'fixed20 selection changed')
    for c in cases:
        c['kind']='initial';c['prefix']=[];c['item']['kind']='initial'
        target=c['item']['initial']['field_name']
        c['snapshot']=dict(state={'availability_ledger':[dict(scope='record',target=target,type='field_name',status='obtained',
             skill='initial-input',source='public-input',value=target)]},counts={},lineage=dict(evidence=0,candidates=0,selection=0),
             comparison_entries={},prefix_steps=[],feedback=None,target_exact=True,prefix_calls=0)
        captured=[]
        def intercept(messages):
            captured.append(deepcopy(messages));raise TargetReached()
        try:rollout(c['item'],inputs['study'],intercept,c['snapshot'])
        except TargetReached:pass
        require(len(captured)==1,'new initial target must be reached exactly')
        c['messages']=captured[0]
    return inputs['study'],cases,rollout

def preflight():
    study,cases,rollout=prepare()
    from program_loop.validation import parse_program
    engine=load_engine(E2);ctrl=engine.old.c.controller(lambda x:'{}',study)
    plan=json.dumps({'program':[{'tool':'GET_FIELD_METADATA','arguments':{}} for _ in range(13)]})
    require(len(parse_program(plan,ctrl.bridge.catalog,{},16).steps)==13,'13-step native Program should parse')
    try:parse_program(plan,ctrl.bridge.catalog,{},6)
    except ValueError:pass
    else:raise ValueError('negative six-step cap test failed')
    for c in cases:
        trace=rollout(c['item'],study,lambda messages:'{}',c['snapshot'])
        require(len(trace['planner_attempts'])==10,'actual native runtime still caps planning at six')
    return dict(cases=20,initial_cases=20,first_targets_captured=True,native13step_accepted=True,
        native10attempts_executed=True,budget=limits_remaining(1,0),model_calls=0,labels_read=False)

class Adapter:
    def __init__(self,backend):self.backend=backend;self.last_generation_metadata={};self.last_call_evidence={}
    def generate(self,messages,arm,stage,destination):
        try:return self.backend.generate(rewrite(messages),'B',stage,destination)
        finally:
            self.last_generation_metadata=deepcopy(self.backend.last_generation_metadata)
            self.last_call_evidence=deepcopy(self.backend.last_call_evidence)

def main():
    run=ROOT/'runs';run.mkdir(exist_ok=False)
    admission=json.loads((ROOT/'admission.json').read_bytes())
    exclusive_json(run/'started.json',dict(pid=os.getpid(),time_unix=time.time(),admission=admission))
    exclusive_json(run/'native-preflight.json',preflight())
    study,cases,rollout=prepare();backend=ModelBackend(E2,admission);backend._ensure_loaded();backend._select('B')
    exclusive_json(run/'model-identity.json',backend.identity)
    rows=[]
    for index,case in enumerate(cases,1):
        folder=run/f'native-{index:02d}-unified';folder.mkdir()
        row=dict(index=index,source_id=case['source_id'],condition='unified',kind='initial',legal_completion=False)
        bridge=ContextBackend(case,Adapter(backend),'B',folder,admission)
        try:
            check_deadline(admission)
            trace=rollout(case['item'],study,bridge,case['snapshot'])
            exclusive_json(folder/'trace.json',trace)
            replay=CPUReplay(bridge.records);recovered=rollout(case['item'],study,replay,case['snapshot'])
            exact=raw_bytes(trace)==raw_bytes(recovered) and replay.position==len(replay.records)
            require(exact,'native replay differs')
            row.update(legal_completion=trace['terminated'],prediction=trace['final_result'],stop_reason=trace['stop_reason'],
                       native_replay_exact=exact,planner_attempts=len(trace['planner_attempts']),executed_steps=trace['new_steps'])
        except BaseException as exc:
            if isinstance(exc,(KeyboardInterrupt,SystemExit)):raise
            row.update(error_type=type(exc).__name__,error=str(exc))
        row.update(model_calls=len(bridge.records),actual_target_exact=bridge.target_exact)
        exclusive_json(folder/'result.json',row);rows.append(row)
    exclusive_json(run/'completion.json',dict(native_denominator=20,fixed_denominator=0,native_results=rows,
        training_updates=0,teacher_api_calls=0,time_unix=time.time(),budget=limits_remaining(1,0)))
    return 0

if __name__=='__main__':
    if sys.argv[1:]==['--preflight']:print(json.dumps(preflight()))
    else:raise SystemExit(main())
