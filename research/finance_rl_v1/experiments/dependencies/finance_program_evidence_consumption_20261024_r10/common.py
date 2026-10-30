import hashlib,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PARENT=ROOT.parent/'finance_program_student_fresh162_20261022_r5'
BASELINE=ROOT.parent/'finance_program_recovery_source_scope_20261024_r9'
ORIGINAL=ROOT.parent/'finance_program_recovery_feedback_20261024_r4'
CAMPAIGN=ROOT
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def put(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,separators=(',',':'));stream.write('\n');stream.flush();os.fsync(stream.fileno())
def verify():
    lock=json.loads((ROOT/'freeze.json').read_bytes())
    for row in lock['files']:
        path=ROOT/row['path'];assert not path.is_symlink() and path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],str(path)
    for row in lock['dependencies']:
        path=ROOT.parent/row['path'];assert not path.is_symlink() and path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],str(path)
    return lock
