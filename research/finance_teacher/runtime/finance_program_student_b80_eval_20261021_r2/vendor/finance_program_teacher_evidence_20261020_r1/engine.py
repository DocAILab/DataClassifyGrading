"""Isolated per-run wrapper; old frozen whole loop and native guards stay intact."""
from copy import deepcopy
import importlib.util
from pathlib import Path

ROOT=Path(__file__).resolve().parent
def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
wide=load('evidence_wide_engine',ROOT.parent/'finance_program_teacher_widebudget_20261020_r1/engine.py')
boundary=load('evidence_boundary_impl',ROOT/'boundary.py')
old=wide.old

def rollout(item,study,generate,snapshot):
    guarded=boundary.Guard(generate);factory=wide.old.c.controller
    def controller(*args,**kwargs):
        ctrl=factory(*args,**kwargs);native_step=ctrl.bridge.step
        def step(call,record,state,counts,lineage,failures=None):
            guarded.record,guarded.state=record,state
            before=boundary.evidence_view(record,state);n=len(guarded.audits)
            actual,completion=boundary.complete_optional(call,record,state,ctrl.bridge.catalog)
            result=native_step(actual,record,state,counts,lineage,failures)
            after=boundary.evidence_view(record,state)
            result['planner_call']=call.to_dict();result['input_completion']=completion
            result['evidence_audits']=deepcopy(guarded.audits[n:])
            result['new_raw_fact_ids']=[k for k,v in after['facts'].items() if k not in before['facts'] or v['value']!=before['facts'][k]['value']]
            result['new_model_interpretation_keys']=[k for k in after['hypotheses'] if k not in before['hypotheses']]
            return result
        ctrl.bridge.step=step;return ctrl
    wide.old.c.controller=controller
    try:
        trace=wide.rollout(item,study,guarded,snapshot)
        trace['evidence_audits']=deepcopy(guarded.audits)
        trace['evidence_protocol']='acquired-evidence-boundary-v1'
        return trace
    finally:wide.old.c.controller=factory
