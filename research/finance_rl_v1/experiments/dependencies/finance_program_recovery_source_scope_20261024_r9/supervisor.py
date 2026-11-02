"""Round2 version2: inherit R8 absolute clock, ledger and global ownership."""
import fcntl,importlib.util,json,os,subprocess,sys,time
from pathlib import Path
from common import ROOT,PARENT,ORIGINAL,PREVIOUS,CAMPAIGN,put,verify,sha
sys.path.append(str(PARENT))
from resource_profile import candidates,recheck
from runtime_safety import guard_resources
spec=importlib.util.spec_from_file_location('completed_training_supervision',PARENT/'supervisor.py')
prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
snapshot,lock_combo=prior.snapshot,prior.lock_combo
resume_spec=importlib.util.spec_from_file_location('verified_clock_resume',ROOT.parent/'finance_program_recovery_feedback_20261024_r5/campaign.py')
resume_module=importlib.util.module_from_spec(resume_spec);resume_spec.loader.exec_module(resume_module)
resume_admission=resume_module.resume_admission
PY='/home/zhubh/finance_program_grpo_full_migrated_20261027_r3/dependencies/finance_program_5090_smoke_20261010/env/bin/python'
def prior_live():
    roots=(PARENT,ORIGINAL,ORIGINAL.parent/'finance_program_recovery_feedback_20261024_r5',PREVIOUS,CAMPAIGN)
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            cmd=(p/'cmdline').read_bytes();state=(p/'stat').read_text().split(') ',1)[1].split()[0]
            if b'-c' not in cmd and any(str(r).encode() in cmd for r in roots) and state!='Z':return True
        except (FileNotFoundError,ProcessLookupError,PermissionError):pass
    return False
def history_closed():
    for name in ('r4','r5','r6'):
        old=ROOT.parent/('finance_program_recovery_feedback_20261024_'+name)
        assert json.loads((old/'launch/exit.json').read_bytes())['actual_returncode']==0
    assert len(list((ORIGINAL/'campaign/call-claims').glob('*.json')))==429
    assert len(list((ORIGINAL/'campaign/version-claims').glob('*.json')))==3
    assert json.loads((CAMPAIGN/'launch/exit.json').read_bytes())['actual_returncode']==0
    completion=json.loads((CAMPAIGN/'runs/completion.json').read_bytes())
    assert completion['stage']=='gate_not_met' and len(completion['rows'])==20
    assert len(list((CAMPAIGN/'campaign/call-claims').glob('*.json')))==160
    assert len(list((CAMPAIGN/'campaign/version-claims').glob('*.json')))==1
def main():
    verify();history_closed();assert not prior_live()
    folder=ROOT/'supervision';folder.mkdir(exist_ok=False);handles=[]
    lock=(ROOT/'writer.lock').open('a+');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    global_lock=(ORIGINAL/'campaign/version-writer.lock').open('a+');fcntl.flock(global_lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    campaign_lock=(CAMPAIGN/'campaign/version-writer.lock').open('a+');fcntl.flock(campaign_lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    put(folder/'started.json',dict(pid=os.getpid(),time_unix=time.time(),round=2,global_version=5,prior_calls=589))
    try:
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
        first=json.loads((CAMPAIGN/'admission.json').read_bytes());started=time.time()
        admission=resume_admission(first,admitted,now=started,campaign_root=str(CAMPAIGN))
        admission.update(round=2,version=2,prior_round_claimed=429)
        put(ROOT/'admission.json',admission)
        put(CAMPAIGN/'campaign/version-claims/02.json',dict(root=ROOT.name,version=2,global_version=5,freeze_sha256=sha(ROOT/'freeze.json'),started_unix=started))
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(admission['gpu_uuids']),PYTHONUTF8='1',TOKENIZERS_PARALLELISM='false')
        with (folder/'worker.stdout.log').open('xb') as out,(folder/'worker.stderr.log').open('xb') as err:
            seconds=int(admission['gpu_deadline_unix']-time.time())-20
            command=['/usr/bin/timeout','--signal=TERM','--kill-after=20s',str(seconds)+'s',PY,'-B',str(ROOT/'runner.py')]
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=out,stderr=err,start_new_session=True,pass_fds=tuple(h.fileno() for h in handles)+(global_lock.fileno(),campaign_lock.fileno()))
            put(folder/'worker-start.json',dict(pid=process.pid,command=command,time_unix=time.time(),inherited_gpu_locks=len(handles),inherited_campaign_locks=2));code=process.wait()
        put(folder/'worker-exit.json',dict(actual_returncode=code,time_unix=time.time()))
        if code!=0:raise RuntimeError('worker actual exit '+str(code))
        verify();put(folder/'completion.json',dict(actual_returncode=0,time_unix=time.time()));return 0
    except BaseException as exc:
        put(folder/'failure.json',dict(error_type=type(exc).__name__,error=str(exc),time_unix=time.time()));raise
    finally:
        for h in handles:h.close()
        campaign_lock.close();global_lock.close();lock.close()
def watch():
    folder=ROOT/'launch';folder.mkdir(exist_ok=False)
    with (folder/'stdout.log').open('xb') as out,(folder/'stderr.log').open('xb') as err:
        process=subprocess.Popen([PY,'-B',str(ROOT/'supervisor.py')],cwd=ROOT,stdout=out,stderr=err)
        put(folder/'started.json',dict(watcher_pid=os.getpid(),supervisor_pid=process.pid,time_unix=time.time()));code=process.wait()
    put(folder/'exit.json',dict(actual_returncode=code,time_unix=time.time()));return code
if __name__=='__main__':raise SystemExit(watch() if sys.argv[1:]==['--watch'] else main())
