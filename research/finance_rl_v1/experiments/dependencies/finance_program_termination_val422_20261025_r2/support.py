"""New full-validation paths; original N100 imports remain bound to their roots."""
import hashlib,importlib.util,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent/'finance_program_termination_first_20261025_r2'
N100=ROOT.parent/'finance_program_student_n100_20261024_r2'
TRAIN=ROOT.parent/'finance_program_student_fresh162_20261022_r5'
VAL=ROOT.parent/'finance_program_student_val422_eval_20261017_r1'
ADAPTER_SHA='6d78e440ab0c9749427e002d54fd94cca06dea77c31b63ae3628b48fdbec4e64'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def put(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,separators=(',',':'),allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())


def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def native():
    sys.path.insert(0,str(N100))
    import eval_runtime
    return eval_runtime


def verify():
    lock=json.loads((ROOT/'freeze.json').read_bytes())
    for row in lock['files']:
        p=ROOT/row['path'];assert not p.is_symlink() and p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],str(p)
    for row in lock['dependencies']:
        p=ROOT.parent/row['path'];assert not p.is_symlink() and p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],str(p)
    return lock


def require_parent():
    path=ROOT/'reviews/stop100-proof.json'
    proof=json.loads(path.read_bytes())
    assert proof['user_requested_stop'] and proof['all_owned_writers_stopped'] and proof['partial_preserved']
    assert proof['worker_actual_returncode']==-2 and proof['launch_actual_returncode']==1
    return dict(user_cancelled_prior100=True,partial_preserved=True,stop100_proof_sha256=sha(path))
