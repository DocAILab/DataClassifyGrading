"""Unique shared-resource admission, real process exits, bounded eight-hour evaluation."""
import fcntl,importlib.util,json,os,subprocess,sys,time
from pathlib import Path
from common import ROOT,PARENT,put,verify
sys.path.append(str(PARENT))
from resource_profile import candidates,recheck
from runtime_safety import guard_resources
spec=importlib.util.spec_from_file_location('completed_training_supervision',PARENT/'supervisor.py')
prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
snapshot,lock_combo=prior.snapshot,prior.lock_combo
PY='/home/zhubh/finance_program_grpo_full_migrated_20261027_r3/dependencies/finance_program_5090_smoke_20261010/env/bin/python'
def prior_live():
    prefixes=[str(PARENT).encode(),b'finance_program_recovery_feedback_20261023_r1']
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            cmd=(p/'cmdline').read_bytes();state=(p/'stat').read_text().split(') ',1)[1].split()[0]
            if b'-c' not in cmd and any(s in cmd for s in prefixes) and state!='Z':return True
        except (FileNotFoundError,ProcessLookupError,PermissionError):pass
    return False
def main():
    verify();assert not prior_live(),'old owned writer live'
    folder=ROOT/'supervision';folder.mkdir(exist_ok=False)
    lock=(ROOT/'writer.lock').open('a+');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);handles=[]
    put(folder/'started.json',dict(pid=os.getpid(),time_unix=time.time()))
    try:
        guard_resources(ROOT,preload=True)
        admitted=None
        for combo in candidates(snapshot()):
            handles=lock_combo(combo)
            if handles is None:handles=[];continue
            try:admitted=recheck(snapshot(),combo)
            except ValueError:
                for h in handles:h.close()
                handles=[];continue
            break
        if admitted is None:raise RuntimeError('no safely admissible three-GPU combo; no blind retries')
        old=json.loads((PARENT/'admission.json').read_bytes());started=time.time()
        admission=dict(old,**{k:v for k,v in admitted.items() if k not in old})
        admission.update(admitted,gpu_started_unix=started,gpu_deadline_unix=started+8*3600,evaluation_only=True,training_root=str(PARENT))
        put(ROOT/'admission.json',admission)
        put(ROOT/'campaign/admission.json',dict(gpu_started_unix=started,gpu_deadline_unix=admission['gpu_deadline_unix'],maximum_versions=3,maximum_calls=5040))
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(admission['gpu_uuids']),PYTHONUTF8='1',TOKENIZERS_PARALLELISM='false')
        with (folder/'worker.stdout.log').open('xb') as out,(folder/'worker.stderr.log').open('xb') as err:
            command=['/usr/bin/timeout','--signal=TERM','--kill-after=20s','28780s',PY,'-B',str(ROOT/'runner.py')]
            worker=subprocess.Popen(command,cwd=ROOT,env=env,stdout=out,stderr=err,start_new_session=True,pass_fds=tuple(h.fileno() for h in handles))
            put(folder/'worker-start.json',dict(pid=worker.pid,command=command,time_unix=time.time(),inherited_locks=len(handles)))
            code=worker.wait()
        put(folder/'worker-exit.json',dict(actual_returncode=code,time_unix=time.time()))
        if code!=0:raise RuntimeError('worker real exit '+str(code))
        verify();put(folder/'completion.json',dict(actual_returncode=0,time_unix=time.time()))
        return 0
    except BaseException as exc:
        put(folder/'failure.json',dict(error_type=type(exc).__name__,error=str(exc),time_unix=time.time()));raise
    finally:
        for h in handles:h.close()
        lock.close()
def watch():
    folder=ROOT/'launch';folder.mkdir(exist_ok=False)
    with (folder/'stdout.log').open('xb') as out,(folder/'stderr.log').open('xb') as err:
        p=subprocess.Popen([PY,'-B',str(ROOT/'supervisor.py')],cwd=ROOT,stdout=out,stderr=err)
        put(folder/'started.json',dict(watcher_pid=os.getpid(),supervisor_pid=p.pid,time_unix=time.time()))
        code=p.wait()
    put(folder/'exit.json',dict(actual_returncode=code,time_unix=time.time()));return code
if __name__=='__main__':raise SystemExit(watch() if sys.argv[1:]==['--watch'] else main())
