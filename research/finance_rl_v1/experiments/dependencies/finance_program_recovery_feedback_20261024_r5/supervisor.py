"""Second independent version, same original campaign and actual shared GPU locks."""
import fcntl,importlib.util,json,os,subprocess,sys,time
from pathlib import Path
from common import ROOT,PARENT,CAMPAIGN,put,verify,sha
from campaign import resume_admission
sys.path.append(str(PARENT))
from resource_profile import candidates,recheck
from runtime_safety import guard_resources
spec=importlib.util.spec_from_file_location('prior_completed_supervision',PARENT/'supervisor.py')
prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
snapshot,lock_combo=prior.snapshot,prior.lock_combo
PY='/home/zhubh/finance_program_grpo_full_migrated_20261027_r3/dependencies/finance_program_5090_smoke_20261010/env/bin/python'
def prior_live():
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            cmd=(p/'cmdline').read_bytes();state=(p/'stat').read_text().split(') ',1)[1].split()[0]
            if b'-c' not in cmd and any(str(r).encode() in cmd for r in (PARENT,CAMPAIGN)) and state!='Z':return True
        except (FileNotFoundError,ProcessLookupError,PermissionError):pass
    return False
def main():
    verify();assert not prior_live();assert json.loads((CAMPAIGN/'launch/exit.json').read_bytes())['actual_returncode']==0
    folder=ROOT/'supervision';folder.mkdir(exist_ok=False);handles=[]
    lock=(ROOT/'writer.lock').open('a+');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    campaign_lock=(CAMPAIGN/'campaign/version-writer.lock').open('a+');fcntl.flock(campaign_lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    put(folder/'started.json',dict(pid=os.getpid(),time_unix=time.time(),version=2))
    try:
        first=json.loads((CAMPAIGN/'admission.json').read_bytes())
        guard_resources(ROOT,preload=True);admitted=None
        for combo in candidates(snapshot()):
            handles=lock_combo(combo)
            if handles is None:handles=[];continue
            try:admitted=recheck(snapshot(),combo)
            except ValueError:
                for h in handles:h.close()
                handles=[];continue
            break
        if admitted is None:raise RuntimeError('no safely admissible three-GPU combination')
        admission=resume_admission(first,admitted,now=time.time(),campaign_root=str(CAMPAIGN))
        put(ROOT/'admission.json',admission)
        versions=CAMPAIGN/'campaign/version-claims';versions.mkdir(exist_ok=True)
        adopted=dict(root=CAMPAIGN.name,version=1,gate_sha256=sha(CAMPAIGN/'runs/gate-result.json'))
        if not (versions/'01.json').exists():put(versions/'01.json',adopted)
        else:assert json.loads((versions/'01.json').read_bytes())==adopted
        assert len(list(versions.glob('*.json')))==1
        put(versions/'02.json',dict(root=ROOT.name,version=2,started_unix=time.time(),freeze_sha256=sha(ROOT/'freeze.json')))
        seconds=int(admission['gpu_deadline_unix']-time.time())-20
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(admission['gpu_uuids']),PYTHONUTF8='1',TOKENIZERS_PARALLELISM='false')
        with (folder/'worker.stdout.log').open('xb') as out,(folder/'worker.stderr.log').open('xb') as err:
            command=['/usr/bin/timeout','--signal=TERM','--kill-after=20s',str(seconds)+'s',PY,'-B',str(ROOT/'runner.py')]
            p=subprocess.Popen(command,cwd=ROOT,env=env,stdout=out,stderr=err,start_new_session=True,pass_fds=tuple(h.fileno() for h in handles)+(campaign_lock.fileno(),))
            put(folder/'worker-start.json',dict(pid=p.pid,command=command,time_unix=time.time(),inherited_gpu_locks=len(handles),campaign_lock_inherited=True));code=p.wait()
        put(folder/'worker-exit.json',dict(actual_returncode=code,time_unix=time.time()))
        if code!=0:raise RuntimeError('worker actual exit '+str(code))
        verify();put(folder/'completion.json',dict(actual_returncode=0,time_unix=time.time()));return 0
    except BaseException as exc:
        put(folder/'failure.json',dict(error_type=type(exc).__name__,error=str(exc),time_unix=time.time()));raise
    finally:
        for h in handles:h.close()
        campaign_lock.close();lock.close()
def watch():
    folder=ROOT/'launch';folder.mkdir(exist_ok=False)
    with (folder/'stdout.log').open('xb') as out,(folder/'stderr.log').open('xb') as err:
        p=subprocess.Popen([PY,'-B',str(ROOT/'supervisor.py')],cwd=ROOT,stdout=out,stderr=err)
        put(folder/'started.json',dict(watcher_pid=os.getpid(),supervisor_pid=p.pid,time_unix=time.time()));code=p.wait()
    put(folder/'exit.json',dict(actual_returncode=code,time_unix=time.time()));return code
if __name__=='__main__':raise SystemExit(watch() if sys.argv[1:]==['--watch'] else main())
