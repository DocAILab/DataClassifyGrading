"""New experiment paths; previous runtime and checkpoints are read-only."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
DEPS=Path(os.environ.get('FINANCE_DEPENDENCY_HOME',str(ROOT/'dependencies')))
TRAIN=DEPS/'finance_program_student_fresh162_20261022_r5'
N100=DEPS/'finance_program_student_n100_20261024_r2'
BASE=DEPS/'finance_program_termination_first_20261025_r2'
VAL=DEPS/'finance_program_termination_val422_20261025_r2'
ADAPTER_SHA='6d78e440ab0c9749427e002d54fd94cca06dea77c31b63ae3628b48fdbec4e64'

def raw(value):return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')
def digest(value):return hashlib.sha256(raw(value)).hexdigest()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def put(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f:f.write(raw(value)+b'\n');f.flush();os.fsync(f.fileno())
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def native():
    if str(N100) not in sys.path:sys.path.insert(0,str(N100))
    if str(BASE) not in sys.path:sys.path.append(str(BASE))
    if str(ROOT) in sys.path:sys.path.remove(str(ROOT))
    sys.path.insert(0,str(ROOT))
    import eval_runtime
    return eval_runtime
def training():
    native()
    if str(TRAIN) not in sys.path:sys.path.append(str(TRAIN))
    return load('rl_original_training_backend',TRAIN/'model_backend.py')
def verify():
    lock=json.loads((ROOT/'freeze.json').read_bytes())
    for base,key in [(ROOT,'files'),(DEPS,'dependencies')]:
        for row in lock[key]:
            p=base/row['path']
            if p.is_symlink() or not p.is_file() or sha(p)!=row['sha256']:raise ValueError('frozen identity changed: '+str(p))
    return lock

