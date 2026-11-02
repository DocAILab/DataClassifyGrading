"""Unmodified R9 runtime with this experiment's isolated initial states."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
from contract import ROOT
E2=ROOT.parent/'finance_program_student_b80_eval_20261021_r2'
R9=ROOT.parent/'finance_program_skill_recovery20_20261022_r9'
sys.path.insert(0,str(E2))
from paired_eval import ContextBackend,CPUReplay,TargetReached,TechnicalFailure,BudgetExhausted,load_engine
from prepare_runtime import raw_bytes,exclusive_json,require
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module
interface=load('fresh162_exact_R9_interface',R9/'interface.py')
runtime=load('fresh162_exact_R9_runtime',R9/'runtime.py')
def prepare():
    inputs=json.loads((ROOT/'eval-inputs.json').read_bytes());rollout=runtime.build(load_engine(E2));cases=deepcopy(inputs['cases'])
    for case in cases:
        messages=[]
        def capture(m):messages.append(deepcopy(m));raise TargetReached()
        try:rollout(case['item'],inputs['study'],capture,case['snapshot'])
        except TargetReached:pass
        require(len(messages)==1,'one actual initial Planner target required');case['messages']=messages[0]
    return inputs['study'],cases,rollout
def preflight():
    study,cases,rollout=prepare()
    for case in cases:
        bridge=ContextBackend(case,None,None,Path('.'),probe=True)
        try:rollout(case['item'],study,bridge,case['snapshot'])
        except TargetReached:pass
        require(bridge.target_exact and not bridge.records,'CPU initial target exact')
    engine=load_engine(E2)
    from program_loop.validation import parse_program
    ctrl=engine.old.c.controller(lambda m:'{}',study)
    plan=json.dumps(dict(program=[dict(tool='GET_FIELD_METADATA',arguments={}) for _ in range(13)]))
    require(len(parse_program(plan,ctrl.bridge.catalog,{},16).steps)==13,'no six-step cap')
    trace=rollout(cases[0]['item'],study,lambda m:'{}',cases[0]['snapshot'])
    require(len(trace['planner_attempts'])==10,'native10 Planner budget')
    return dict(cases=40,exact_first_targets=40,slots=80,native13steps=True,native10planner=True,new_model_calls=0)
