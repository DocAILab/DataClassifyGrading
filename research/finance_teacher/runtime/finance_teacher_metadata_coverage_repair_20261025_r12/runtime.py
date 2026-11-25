"""Exact old runtime until deterministic tool lookup; new data only after real GET."""
import importlib.util,sys,types
from pathlib import Path
import coverage_pool
ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'finance_teacher_failedcases_retry_20261024_r9'
sys.path.insert(0,str(OLD))
spec=importlib.util.spec_from_file_location('r12_pinned_R9_runtime',OLD/'runtime.py');previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
native_context,CommonActor=previous.native_context,previous.CommonActor
def build(engine,scoped,stopped,bundle=None):
 if bundle is None:bundle=coverage_pool.read(ROOT/'coverage-bundle-01.json')
 rollout=previous.build(engine,scoped,stopped);namespace=dict(rollout.__globals__);old=namespace['old'];oldcopy=types.SimpleNamespace(**vars(old));ccopy=types.SimpleNamespace(**vars(old.c));original=ccopy.controller
 # The original controller, catalog and acquired-provider binder are constructed
 # unchanged. Only the deterministic metadata acquisition backend is substituted.
 rows=coverage_pool.copy.deepcopy(bundle['expanded_rows']);scopes=coverage_pool.copy.deepcopy(bundle['expanded_scopes'])
 def create(*args,**kwargs):
  controller=original(*args,**kwargs);controller.bridge.executor.metadata_lookup=coverage_pool.scope.SourceScopedMetadata(rows,scopes);return controller
 ccopy.controller=create;oldcopy.c=ccopy;namespace['old']=oldcopy
 return types.FunctionType(rollout.__code__,namespace,rollout.__name__,rollout.__defaults__,rollout.__closure__)
