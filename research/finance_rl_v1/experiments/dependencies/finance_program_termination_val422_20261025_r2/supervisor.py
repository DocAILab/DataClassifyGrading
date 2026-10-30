"""Wait-free safe admission after100 settlement, one bounded full422 worker."""
import fcntl,json,os,subprocess,sys,time
from pathlib import Path
from support import ROOT,BASE,TRAIN,native,load,put,verify,require_parent
native();sys.path.append(str(TRAIN))
from resource_profile import candidates,recheck,PINNED_UUIDS
from runtime_safety import guard_resources
old=load('full422_shared_resource_supervisor',TRAIN/'supervisor.py')
PY='/home/zhubh/finance_program_grpo_full_migrated_20261027_r3/dependencies/finance_program_5090_smoke_20261010/env/bin/python'


def main():
    verify();parent=require_parent()
    proof=json.loads((ROOT/'reviews/preflight.json').read_bytes())
    assert proof['passed'] and proof['exact_initial_targets']==422 and proof['new_model_calls']==0
    run=ROOT/'supervision';run.mkdir(exist_ok=False);handles=[]
    lock=(ROOT/'writer.lock').open('a+');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    put(run/'started.json',dict(pid=os.getpid(),time_unix=time.time(),parent=parent))
    try:
        for p in Path('/proc').iterdir():
            if not p.name.isdigit():continue
            try:
                cmd=(p/'cmdline').read_bytes();state=(p/'stat').read_text().split(') ',1)[1].split()[0]
                if str(BASE).encode() in cmd and b'-c' not in cmd and state!='Z':raise RuntimeError('prior100 writer remains live')
            except (FileNotFoundError,ProcessLookupError,PermissionError):pass
        guard_resources(ROOT,preload=True);admitted=None
        for combo in candidates(old.snapshot()):
            if combo['gpu_indices']!=[0,2,3] or combo['gpu_uuids']!=[PINNED_UUIDS[i] for i in (0,2,3)]:continue
            handles=old.lock_combo(combo)
            if handles is None:handles=[];continue
            try:admitted=recheck(old.snapshot(),combo)
            except ValueError:
                for h in handles:h.close()
                handles=[];continue
            break
        if admitted is None:raise RuntimeError('full422 shared GPU locks/free memory unavailable; no unsafe launch')
        admission=dict(json.loads((TRAIN/'admission.json').read_bytes()),**admitted)
        start=time.time();admission.update(gpu_started_unix=start,gpu_deadline_unix=start+8*3600,evaluation_only=True,
                                            evaluation_root=ROOT.name,N_only=True,source_denominator=422,model_call_limit=17724,
                                            new_window_reason='user_authorized_full_validation_after100_20261025',parent=parent)
        put(ROOT/'admission.json',admission)
        command=['/usr/bin/timeout','--signal=TERM','--kill-after=20s',str(8*3600-20)+'s',PY,'-B',str(ROOT/'runner.py')]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(admission['gpu_uuids']),PYTHONUTF8='1',TOKENIZERS_PARALLELISM='false')
        with (run/'worker.stdout.log').open('xb') as out,(run/'worker.stderr.log').open('xb') as err:
            p=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True,
                               pass_fds=tuple(h.fileno() for h in handles+[lock]))
            put(run/'worker-start.json',dict(pid=p.pid,command=command,time_unix=time.time(),inherited_locks=4));code=p.wait()
        put(run/'worker-exit.json',dict(actual_returncode=code,time_unix=time.time()))
        if code!=0:raise RuntimeError('full422 actual exit '+str(code)+'; no blind retry')
        verify();c=json.loads((ROOT/'runs/completion.json').read_bytes());assert len(c['rows'])==422
        put(run/'completion.json',dict(actual_returncode=0,time_unix=time.time()));return 0
    except BaseException as exc:
        put(run/'failure.json',dict(error_type=type(exc).__name__,error=str(exc),time_unix=time.time()));raise
    finally:
        for h in handles:h.close()
        lock.close()


def watch():
    folder=ROOT/'launch';folder.mkdir(exist_ok=False)
    with (folder/'stdout.log').open('xb') as out,(folder/'stderr.log').open('xb') as err:
        p=subprocess.Popen([PY,'-B',str(ROOT/'supervisor.py')],cwd=ROOT,stdout=out,stderr=err)
        put(folder/'started.json',dict(watcher_pid=os.getpid(),supervisor_pid=p.pid,time_unix=time.time()));code=p.wait()
    put(folder/'exit.json',dict(actual_returncode=code,time_unix=time.time()));return code


if __name__=='__main__':raise SystemExit(watch() if '--watch' in sys.argv else main())
