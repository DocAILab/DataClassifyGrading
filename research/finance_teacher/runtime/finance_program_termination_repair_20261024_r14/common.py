import hashlib,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PARENT=ROOT.parent/'finance_program_student_fresh162_20261022_r5'
PREVIOUS=ROOT.parent/'finance_program_evidence_consumption_20261024_r11'
BASELINE=ROOT.parent/'finance_program_recovery_source_scope_20261024_r9'
N100=ROOT.parent/'finance_program_student_n100_20261024_r2'
CAMPAIGN=ROOT
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,separators=(',',':'));f.write('\n');f.flush();os.fsync(f.fileno())
def verify():
 lock=json.loads((ROOT/'freeze.json').read_bytes())
 for key,base in (('files',ROOT),('dependencies',ROOT.parent)):
  for row in lock[key]:
   p=base/row['path'];assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],str(p)
 return lock
