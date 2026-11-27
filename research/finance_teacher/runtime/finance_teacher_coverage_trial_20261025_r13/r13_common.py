"""Unique execution helpers. Receipts immutable; never store API credentials."""
import hashlib,importlib.util,json,os,sys,time,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parent
R10=ROOT.parent/'finance_teacher_200_controller_20261024_r10'
R11=ROOT.parent/'finance_teacher_program75_20261025_r11'
R12=ROOT.parent/'finance_teacher_metadata_coverage_repair_20261025_r12'
R9=ROOT.parent/'finance_teacher_failedcases_retry_20261024_r9'
GLOBAL=ROOT.parent/'finance_teacher_program_coldstart_20261024_r1/program-writer.lock'
def read(p):return json.loads(Path(p).read_bytes())
def sha(b):return hashlib.sha256(b).hexdigest()
def binding(p):
 p=Path(p).resolve();b=p.read_bytes();return dict(path=str(p),sha256=sha(b),bytes=len(b))
def exact(row):
 if binding(row['path'])!=row:raise ValueError('original file SHA or size changed')
def put(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():raise FileExistsError('immutable receipt exists')
 tmp=p.with_name(p.name+'.'+uuid.uuid4().hex+'.partial')
 with tmp.open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
 os.rename(tmp,p)
def load(name,p):
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def verify_freeze():
 for row in read(ROOT/'freeze-01.json')['files']:exact(row)
 return sha((ROOT/'freeze-01.json').read_bytes())
