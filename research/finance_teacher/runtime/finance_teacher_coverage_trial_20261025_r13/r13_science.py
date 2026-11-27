"""Collection and independent native replay share exact frozen R12 lookup."""
import sys
from pathlib import Path
from r13_common import *
_runtime=None
def get_runtime():
 global _runtime
 if _runtime is None:
  sys.path.insert(0,str(R12));sys.path.insert(0,str(R9))
  pool=load('coverage_pool',R12/'coverage_pool.py')
  _runtime=load('r13_explicit_R12_runtime',R12/'runtime.py')
  if Path(_runtime.coverage_pool.__file__).resolve()!=R12/'coverage_pool.py':raise ValueError('coverage module collision')
 return _runtime
def load_replay():
 rt=get_runtime();saved=sys.modules.get('runtime');sys.modules['runtime']=rt
 try:
  module=load('r13_independent_R4_replay',ROOT.parent/'finance_teacher_program_coldstart_20261024_r4/verify_program.py')
 finally:
  if saved is None:sys.modules.pop('runtime',None)
  else:sys.modules['runtime']=saved
 module.build=rt.build;module.CommonActor=rt.CommonActor;module.native_context=rt.native_context
 return module
def contract():return dict(R12_freeze=binding(R12/'freeze-01.json'),lookup_bundle=binding(R12/'coverage-bundle-01.json'))
def require_contract(proof):
 if proof.get('runtime_contract')!=contract():raise ValueError('proof is not this R12 317-row contract')
 for row in proof['runtime_contract'].values():exact(row)
 return proof
def verify_source(target,receipt):
 target,receipt=Path(target),Path(receipt)
 rt=get_runtime();v=load('r13_pinned_R9_full_independent_verifier',R9/'verify_retry.py')
 if receipt.exists():require_contract(read(receipt))
 original_load=v.load;original_put=v.put
 def pinned_load(name,path):
  if Path(path).resolve()==ROOT.parent/'finance_teacher_program_coldstart_20261024_r4/verify_program.py':return load_replay()
  return original_load(name,path)
 def bound_put(path,value):
  if Path(path).resolve()==receipt.resolve():value=dict(value,runtime_contract=contract())
  return original_put(path,value)
 v.load=pinned_load;v.put=bound_put
 saved=sys.modules.get('runtime');sys.modules['runtime']=rt
 try:
  v.verify(target,receipt)
  # The pinned verifier returns its pre-put object; the immutable receipt is the
  # authority because bound_put adds this prospective scientific contract.
  return require_contract(read(receipt))
 finally:
  if saved is None:sys.modules.pop('runtime',None)
  else:sys.modules['runtime']=saved
