"""Original fresh162 cases and native contracts, with isolated repaired adapter."""
import importlib.util
import sys
from common import PARENT
sys.path.append(str(PARENT))
spec=importlib.util.spec_from_file_location('consumption_original_eval',PARENT/'eval_runtime.py')
parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
ContextBackend,CPUReplay,raw_bytes=parent.ContextBackend,parent.CPUReplay,parent.raw_bytes
require=parent.require
TechnicalFailure,BudgetExhausted,TargetReached=parent.TechnicalFailure,parent.BudgetExhausted,parent.TargetReached
def prepare():
    study,cases,_=parent.prepare()
    import runtime
    return study,cases,runtime.build(parent.load_engine(parent.E2))
def preflight():
    study,cases,rollout=prepare()
    for case in cases:
        backend=ContextBackend(case,None,None,__import__('pathlib').Path('.'),probe=True)
        try:rollout(case['item'],study,backend,case['snapshot'])
        except TargetReached:pass
        assert backend.target_exact and not backend.records
    return dict(cases=40,exact_first_targets=40,new_model_calls=0)
