"""Per-controller metadata injection, no global vendor or native guard mutation."""
import copy,importlib.util,json,types
from pathlib import Path
from source_scope import SourceScopedMetadata
ROOT=Path(__file__).resolve().parent
def scoped_controller(original,scope_rows):
    private_rows=copy.deepcopy(scope_rows)
    def create(router,study,*args,**kwargs):
        controller=original(router,study,*args,**kwargs)
        controller.bridge.executor.metadata_lookup=SourceScopedMetadata(study['rows'],private_rows)
        return controller
    return create

def build(engine):
    path=ROOT.parent/'finance_program_skill_recovery20_20261022_r9/runtime.py'
    spec=importlib.util.spec_from_file_location('source_scope_unchanged_R9_budget',path)
    previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
    original=previous.build(engine)
    scopes=json.loads((ROOT/'source-scopes.json').read_bytes())['rows']
    namespace=dict(original.__globals__)
    old=namespace['old'];old_copy=types.SimpleNamespace(**vars(old))
    c_copy=types.SimpleNamespace(**vars(old.c))
    c_copy.controller=scoped_controller(old.c.controller,scopes)
    old_copy.c=c_copy;namespace['old']=old_copy
    # A fresh function namespace rather than assignment to a frozen module.
    return types.FunctionType(original.__code__,namespace,original.__name__,original.__defaults__,original.__closure__)
