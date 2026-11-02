"""Unique N100 inference, unchanged campaign deadline and shared resource locks."""
import fcntl,importlib.util,json,os,subprocess,sys,time
from pathlib import Path
from common import ROOT,PARENT,PREVIOUS,BASELINE,ORIGINAL,CAMPAIGN,put,verify,sha
sys.path.append(str(PARENT))
from resource_profile import candidates,recheck,PINNED_UUIDS
from runtime_safety import guard_resources
spec=importlib.util.spec_from_file_location('n100_resource_supervisor',PARENT/'supervisor.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
snapshot,lock_combo=old.snapshot,old.lock_combo
PY='/home/zhubh/finance_program_grpo_full_migrated_20261027_r3/dependencies/finance_program_5090_smoke_20261010/env/bin/python'
def prior_live():
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:
   cmd=(p/'cmdline').read_bytes();state=(p/'stat').read_text().split(') ',1)[1].split()[0]
   if b'-c' not in cmd and state!='Z' and any(str(r).encode() in cmd for r in (ROOT,PREVIOUS,CAMPAIGN,BASELINE,PARENT)):return True
  except (FileNotFoundError,ProcessLookupError,PermissionError):pass
 return False
def history_closed():
 assert json.loads((PREVIOUS/'launch-targeted/exit.json').read_bytes())['actual_returncode']==0
 c=json.loads((PREVIOUS/'runs/targeted-completion.json').read_bytes());assert len(c['rows'])==8 and c['model_calls']==85
 assert not (PREVIOUS/'launch-full').exists()
 assert len(list((CAMPAIGN/'campaign/call-claims').glob('*.json')))==168
 assert len(list((CAMPAIGN/'campaign/version-claims').glob('*.json')))==2
def main():
 verify();folder=ROOT/'supervision-full';folder.mkdir(exist_ok=False);handles=[];locks=[]
 for path in (ROOT/'writer.lock',ORIGINAL/'campaign/version-writer.lock',CAMPAIGN/'campaign/version-writer.lock'):
  h=path.open('a+');fcntl.flock(h.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(h)
 put(folder/'started.json',dict(pid=os.getpid(),time_unix=time.time(),phase='N100',version=3))
 try:
  history_closed();guard_resources(ROOT,preload=True);admitted=None
  for combo in candidates(snapshot()):
   if combo['gpu_indices']!=[0,2,3] or combo['gpu_uuids']!=[PINNED_UUIDS[i] for i in (0,2,3)]:continue
   handles=lock_combo(combo)
   if handles is None:handles=[];continue
   try:admitted=recheck(snapshot(),combo)
   except ValueError:
    for h in handles:h.close()
    handles=[];continue
   break
  if admitted is None:raise RuntimeError('original0/2/3 GPUs not safely admissible')
  first=json.loads((CAMPAIGN/'admission.json').read_bytes());assert first['gpu_started_unix']==1792808412.3172429 and first['gpu_deadline_unix']==1792837212.3172429
  assert first['gpu_deadline_unix']>time.time()+600
  admission=dict(first);admission.update(admitted,interface_root=PREVIOUS.name,evaluation_root=ROOT.name,version=3,N_only=True,fixed_slots=100)
  put(ROOT/'admission.json',admission);put(CAMPAIGN/'campaign/version-claims/03.json',dict(freeze_sha256=sha(ROOT/'freeze.json'),version=3,global_version=8,root=ROOT.name,meaning='user-authorized N100 diagnostic, no new interface hypothesis',time_unix=time.time()))
  env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(admission['gpu_uuids']),PYTHONUTF8='1',TOKENIZERS_PARALLELISM='false')
  seconds=int(first['gpu_deadline_unix']-time.time())-20
  command=['/usr/bin/timeout','--signal=TERM','--kill-after=20s',str(seconds)+'s',PY,'-B',str(ROOT/'runner.py')]
  with (folder/'worker.stdout.log').open('xb') as out,(folder/'worker.stderr.log').open('xb') as err:
   p=subprocess.Popen(command,cwd=ROOT,env=env,stdout=out,stderr=err,start_new_session=True,pass_fds=tuple(h.fileno() for h in handles+locks[1:]));put(folder/'worker-start.json',dict(pid=p.pid,command=command,time_unix=time.time(),inherited_gpu_locks=3,inherited_campaign_locks=2));code=p.wait()
  put(folder/'worker-exit.json',dict(actual_returncode=code,time_unix=time.time()))
  if code!=0:raise RuntimeError('worker actual exit '+str(code))
  verify();put(folder/'completion.json',dict(actual_returncode=0,time_unix=time.time()));return 0
 except BaseException as exc:put(folder/'failure.json',dict(error_type=type(exc).__name__,error=str(exc),time_unix=time.time()));raise
 finally:
  for h in handles+locks:h.close()
def watch():
 folder=ROOT/'launch-full';folder.mkdir(exist_ok=False)
 with (folder/'stdout.log').open('xb') as out,(folder/'stderr.log').open('xb') as err:
  p=subprocess.Popen([PY,'-B',str(ROOT/'supervisor.py')],cwd=ROOT,stdout=out,stderr=err);put(folder/'started.json',dict(watcher_pid=os.getpid(),supervisor_pid=p.pid,time_unix=time.time()));code=p.wait()
 put(folder/'exit.json',dict(actual_returncode=code,time_unix=time.time()));return code
if __name__=='__main__':raise SystemExit(watch() if '--watch' in sys.argv else main())
