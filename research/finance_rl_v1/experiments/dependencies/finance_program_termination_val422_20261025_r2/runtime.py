"""Use the exact current termination program with only the metadata cohort changed."""
from copy import deepcopy
import json,types
from support import ROOT,BASE,N100,load,native


def build(engine,study,scopes):
    old=native()
    module=load('full422_exact_termination_runtime',BASE/'runtime.py')
    rollout=module.build(engine)
    provider=load('full422_same_scoped_provider',ROOT.parent/'finance_program_recovery_source_scope_20261024_r9/source_scope.py')
    binding=load('full422_same_scope_binding',ROOT.parent/'finance_program_evidence_consumption_20261024_r10/scope_binding.py')
    construct=engine.old.c.controller
    def create(generate,local,*args,**kwargs):
        ctrl=construct(generate,local,*args,**kwargs)
        ctrl.bridge.executor.metadata_lookup=provider.SourceScopedMetadata(local['rows'],scopes)
        return binding.install(ctrl)
    namespace=dict(rollout.__globals__)
    old_copy=types.SimpleNamespace(**vars(namespace['old']))
    old_copy.c=types.SimpleNamespace(**vars(old_copy.c));old_copy.c.controller=create
    namespace['old']=old_copy
    return types.FunctionType(rollout.__code__,namespace,rollout.__name__,rollout.__defaults__,rollout.__closure__)


def prepare():
    old=native()
    inputs=json.loads((ROOT/'eval-inputs.json').read_bytes())
    # Preserve the original initial-state canonical construction used by N100.
    inputs=json.loads(json.dumps(inputs,ensure_ascii=False,sort_keys=True,separators=(',',':')))
    study=inputs['study'];cases=deepcopy(inputs['cases'])
    engine=old.parent.load_engine(old.parent.E2)
    scopes=json.loads((ROOT/'source-scopes.json').read_bytes())['rows']
    rollout=build(engine,study,scopes)
    for case in cases:
        messages=[]
        def capture(m):messages.append(deepcopy(m));raise old.TargetReached()
        try:rollout(case['item'],study,capture,case['snapshot'])
        except old.TargetReached:pass
        assert len(messages)==1
        case['messages']=messages[0]
    return study,cases,rollout
