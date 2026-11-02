"""Same native rollout, local binder admission tied to the exact acquired provider."""
import importlib.util
import types
from common import BASELINE
from scope_binding import install
spec=importlib.util.spec_from_file_location('exact_r9_scoped_runtime',BASELINE/'runtime.py')
previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)

def build(engine):
    rollout=previous.build(engine)
    namespace=dict(rollout.__globals__)
    old=namespace['old'];old_copy=types.SimpleNamespace(**vars(old));c_copy=types.SimpleNamespace(**vars(old.c))
    original=c_copy.controller
    def create(*args,**kwargs):return install(original(*args,**kwargs))
    c_copy.controller=create;old_copy.c=c_copy;namespace['old']=old_copy
    return types.FunctionType(rollout.__code__,namespace,rollout.__name__,rollout.__defaults__,rollout.__closure__)
