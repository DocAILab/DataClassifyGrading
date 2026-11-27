"""Private new-target identity adapter; no edits to live/frozen runtime or guards."""
import copy,inspect,sys,types
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from r13_common import read,exact,ROOT,R11
from r13_science import get_runtime
def runtime():return get_runtime()
def bundle():
 from release_inputs import bundle as released_bundle
 return released_bundle()
def build(engine,scoped,stopped,contract):
 rollout=runtime().build(engine,scoped,stopped,{"expanded_rows":contract["lookup_rows"],"expanded_scopes":contract["lookup_scopes"]})
 namespace=dict(rollout.__globals__);old=namespace['old'];oldcopy=types.SimpleNamespace(**vars(old));ccopy=types.SimpleNamespace(**vars(old.c))
 R12=runtime().ROOT/'runtime.py'
 SCOPE=ROOT.parent/'finance_program_recovery_source_scope_20261024_r9/runtime.py'
 counts={'R12':0,'scope':0};memo={}
 def clone(fn):
  if id(fn) in memo:return memo[id(fn)]
  if not fn.__closure__:return fn
  cells=[];path=Path(fn.__code__.co_filename).resolve()
  for name,cell in zip(fn.__code__.co_freevars,fn.__closure__):
   value=cell.cell_contents
   if path==R12 and name=='rows':value=copy.deepcopy(contract['lookup_rows']);counts['R12']+=1
   elif path==R12 and name=='scopes':value=copy.deepcopy(contract['lookup_scopes'])
   elif path==SCOPE and name=='private_rows':value=copy.deepcopy(contract['study_scopes']);counts['scope']+=1
   elif inspect.isfunction(value):value=clone(value)
   cells.append(types.CellType(value))
  result=types.FunctionType(fn.__code__,fn.__globals__,fn.__name__,fn.__defaults__,tuple(cells));result.__kwdefaults__=fn.__kwdefaults__;memo[id(fn)]=result;return result
 ccopy.controller=clone(ccopy.controller)
 if counts!={'R12':1,'scope':1}:raise ValueError('exact frozen metadata factory shape changed')
 oldcopy.c=ccopy;namespace['old']=oldcopy
 return types.FunctionType(rollout.__code__,namespace,rollout.__name__,rollout.__defaults__,rollout.__closure__)
