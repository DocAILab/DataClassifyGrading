"""Delegate to unchanged native parser/type validator, never repair raw outputs."""
import importlib
from pathlib import Path
import sys

E2=Path(__file__).resolve().parent.parent/'finance_program_student_b80_eval_20261021_r2'
sys.path.insert(0,str(E2))
from paired_eval import load_engine,load_inputs

def initialize():
    engine=load_engine(E2)
    inputs,_=load_inputs(E2)
    engine.old.c.controller(lambda messages:'{}',inputs['study'])
    return importlib.import_module('pool_runtime')

def validate_raw(stage,raw,state=None,max_steps=16):
    pool=initialize()
    if stage=='Planner':
        from program_loop.validation import parse_program
        plan=parse_program(raw,pool.pool_catalog('F66'),state or {},max_steps)
        return dict(native_valid=True,planned_steps=len(plan.steps),
                    meaning='native static parse/dependency validity, not execution or correctness')
    if stage!='INTERPRET_CONTEXT_FIELDS':raise ValueError('unapproved replay stage')
    spec=pool.pool_catalog('F66')[stage]
    obs=pool.parse_skill_observation(raw,spec)
    # The context type branch accesses no instance fields; this is the original
    # full native semantic validator, not a newly weakened substitute.
    result=pool.PoolExecutor._validate_semantic(None,obs,spec,{})
    return dict(native_valid=True,data=dict(result.data),summary=result.summary)
