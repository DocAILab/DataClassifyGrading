"""One actual attempt; shared GPU locks, real timeout, immutable exit evidence."""
import fcntl,json,os,subprocess,sys,time
from pathlib import Path
from support import ROOT,TRAIN,put,load,verify,native,sha
from resource_runtime import recheck
PY=str(ROOT/'env/bin/python')

def read(p):return json.loads(Path(p).read_bytes())
def main(attempt):
 verify();native();policy=read(ROOT/'pilot-policy.json');folder=ROOT/f'attempts/{attempt:03d}';request=read(folder/'request.json')
 if not read(ROOT/'reviews/preflight.json')['passed']:raise ValueError('actual CPU preflight missing')
 lock=(ROOT/'writer.lock').open('a+');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);handles=[]
 run=folder/'supervision';run.mkdir(exist_ok=False);put(run/'started.json',dict(pid=os.getpid(),attempt=attempt,time_unix=time.time()))
 try:
  sys.path.append(str(TRAIN));shared=load('full_shared_snapshot',TRAIN/'supervisor.py')
  from runtime_safety import guard_resources
  guard_resources(ROOT,preload=True);anchor=request['candidate'];handles=shared.lock_combo(anchor)
  if handles is None:handles=[];raise RuntimeError('recoverable resource admission lock unavailable')
  admitted=recheck(shared.snapshot(),anchor,policy);now=time.time()
  remaining=min(request['remaining_seconds'],policy['absolute_end_unix']-now)
  if remaining<=60:raise TimeoutError('original global budget exhausted')
  a=dict(read(TRAIN/'admission.json'),**admitted);a.update(gpu_started_unix=now,gpu_deadline_unix=now+remaining,model_call_limit=policy['training_calls'],maximum_optimizer_updates=6,inherited_claims=0,policy_sha256=sha(ROOT/'pilot-policy.json'),attempt_id=attempt)
  put(folder/'admission.json',a)
  env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(a['gpu_uuids']),FINANCE_ATTEMPT_ID=str(attempt),PYTHONUTF8='1',TOKENIZERS_PARALLELISM='false')
  cmd=['/usr/bin/timeout','--signal=TERM','--kill-after=20s',str(int(remaining)-20)+'s',PY,'-B',str(ROOT/'runner.py')]
  with (run/'worker.stdout.log').open('xb') as out,(run/'worker.stderr.log').open('xb') as err:
   child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True,pass_fds=tuple(h.fileno() for h in handles+[lock]))
   put(run/'worker-start.json',dict(pid=child.pid,command=cmd,inherited_locks=4,time_unix=time.time()));code=child.wait()
  put(run/'worker-exit.json',dict(actual_returncode=code,time_unix=time.time(),attempt_id=attempt))
  if code:raise RuntimeError('actual worker exit '+str(code)+'; checkpoint recovery only')
  if not read(ROOT/'runs/completion.json')['completed']:raise ValueError('true training completion missing')
  verify();return 0
 except BaseException as exc:
  put(folder/'supervisor-failure.json',dict(type=type(exc).__name__,error=str(exc),time_unix=time.time()));raise
 finally:
  for h in handles:h.close()
  lock.close()

def watch(attempt):
 folder=ROOT/f'attempts/{attempt:03d}/launch';folder.mkdir(exist_ok=False)
 with (folder/'stdout.log').open('xb') as out,(folder/'stderr.log').open('xb') as err:
  child=subprocess.Popen([PY,'-B',str(ROOT/'attempt_supervisor.py'),'--attempt',str(attempt)],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err)
  put(folder/'started.json',dict(watcher_pid=os.getpid(),supervisor_pid=child.pid,time_unix=time.time()));code=child.wait()
 put(folder/'exit.json',dict(actual_returncode=code,time_unix=time.time()));return code
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--attempt',type=int,required=True);p.add_argument('--watch',action='store_true');a=p.parse_args();raise SystemExit(watch(a.attempt) if a.watch else main(a.attempt))
