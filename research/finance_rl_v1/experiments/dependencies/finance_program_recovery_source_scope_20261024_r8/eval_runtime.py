"""Same native R9 execution, with an instance-scoped metadata address adapter."""
import importlib.util,sys
from common import PARENT
sys.path.append(str(PARENT))
spec=importlib.util.spec_from_file_location('source_scope_original_eval_runtime',PARENT/'eval_runtime.py')
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
        bridge=ContextBackend(case,None,None,__import__('pathlib').Path('.'),probe=True)
        try:rollout(case['item'],study,bridge,case['snapshot'])
        except TargetReached:pass
        assert bridge.target_exact and not bridge.records
    return dict(cases=40,exact_first_targets=40,new_model_calls=0,native_guards_unchanged=True,scoped_lookup_only=True)
