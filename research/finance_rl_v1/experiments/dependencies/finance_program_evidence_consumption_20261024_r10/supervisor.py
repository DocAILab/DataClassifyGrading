"""Bounded new campaign, same UUID/shared headroom protection, no automatic retry."""
import fcntl,importlib.util,json,os,subprocess,sys,time
from pathlib import Path
from common import ROOT,PARENT,BASELINE,ORIGINAL,put,verify,sha
sys.path.append(str(PARENT))
from resource_profile import candidates,recheck,PINNED_UUIDS
from runtime_safety import guard_resources
spec=importlib.util.spec_from_file_location('original_shared_resource_supervision',PARENT/'supervisor.py')
previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
snapshot,lock_combo=previous.snapshot,previous.lock_combo
PY='/home/zhubh/finance_program_grpo_full_migrated_20261027_r3/dependencies/finance_program_5090_smoke_20261010/env/bin/python'
def prior_live():
    roots=(ROOT,BASELINE,PARENT,ROOT.parent/'finance_program_recovery_source_scope_20261024_r8')
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            cmd=(p/'cmdline').read_bytes();state=(p/'stat').read_text().split(') ',1)[1].split()[0]
            if b'-c' not in cmd and state!='Z' and any(str(r).encode() in cmd for r in roots):return True
        except (FileNotFoundError,ProcessLookupError,PermissionError):pass
    return False
def main(phase):
    verify()
    assert json.loads((BASELINE/'launch/exit.json').read_bytes())['actual_returncode']==0
    folder=ROOT/('supervision-'+phase);folder.mkdir(exist_ok=False);handles=[];locks=[]
    for path in (ROOT/'writer.lock',ORIGINAL/'campaign/version-writer.lock',ROOT/'campaign/version-writer.lock'):
        path.parent.mkdir(exist_ok=True,parents=True)
        handle=path.open('a+');fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(handle)
    put(folder/'started.json',dict(pid=os.getpid(),time_unix=time.time(),phase=phase))
    try:
        guard_resources(ROOT,preload=True);admitted=None
        for combo in candidates(snapshot()):
            if combo['gpu_indices']!=[0,2,3] or combo['gpu_uuids']!=[PINNED_UUIDS[i] for i in (0,2,3)]:continue
            handles=lock_combo(combo)
            if handles is None:handles=[];continue
            try:admitted=recheck(snapshot(),combo)
            except ValueError:
                for handle in handles:handle.close()
                handles=[];continue
            break
        if admitted is None:raise RuntimeError('original0/2/3 three-GPU exactUUID/headroom not safely admissible')
        if phase=='targeted':
            old=json.loads((PARENT/'admission.json').read_bytes());started=time.time()
            admission=dict(old);admission.update(admitted,gpu_started_unix=started,gpu_deadline_unix=started+8*3600,
                evaluation_only=True,training_root=str(PARENT),campaign_root=str(ROOT),prior_global_actual_calls=1525)
            put(ROOT/'admission.json',admission)
            put(ROOT/'campaign/admission.json',dict(gpu_started_unix=started,gpu_deadline_unix=admission['gpu_deadline_unix'],maximum_versions=3,maximum_calls=5040,prior_global_actual_calls=1525))
            put(ROOT/'campaign/version-claims/01.json',dict(freeze_sha256=sha(ROOT/'freeze.json'),version=1,global_version=6,time_unix=started))
        else:
            admission=json.loads((ROOT/'admission.json').read_bytes())
            assert admission['gpu_uuids']==admitted['gpu_uuids'] and admission['gpu_deadline_unix']>time.time()+600
            assert json.loads((ROOT/'launch-targeted/exit.json').read_bytes())['actual_returncode']==0
            assert json.loads((ROOT/'reviews/semantic-gate-01.json').read_bytes())['passed'] is True
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(admission['gpu_uuids']),PYTHONUTF8='1',TOKENIZERS_PARALLELISM='false')
        seconds=int(admission['gpu_deadline_unix']-time.time())-20
        command=['/usr/bin/timeout','--signal=TERM','--kill-after=20s',str(seconds)+'s',PY,'-B',str(ROOT/'runner.py'),phase]
        with (folder/'worker.stdout.log').open('xb') as out,(folder/'worker.stderr.log').open('xb') as err:
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=out,stderr=err,start_new_session=True,pass_fds=tuple(h.fileno() for h in handles+locks[1:]))
            put(folder/'worker-start.json',dict(pid=process.pid,command=command,time_unix=time.time(),inherited_gpu_locks=len(handles),inherited_campaign_locks=2));code=process.wait()
        put(folder/'worker-exit.json',dict(actual_returncode=code,time_unix=time.time()))
        if code!=0:raise RuntimeError('worker actual exit '+str(code))
        verify();put(folder/'completion.json',dict(actual_returncode=0,time_unix=time.time()));return 0
    except BaseException as exc:
        put(folder/'failure.json',dict(error_type=type(exc).__name__,error=str(exc),time_unix=time.time()));raise
    finally:
        for handle in handles+locks:handle.close()
def watch(phase):
    folder=ROOT/('launch-'+phase);folder.mkdir(exist_ok=False)
    with (folder/'stdout.log').open('xb') as out,(folder/'stderr.log').open('xb') as err:
        process=subprocess.Popen([PY,'-B',str(ROOT/'supervisor.py'),phase],cwd=ROOT,stdout=out,stderr=err)
        put(folder/'started.json',dict(watcher_pid=os.getpid(),supervisor_pid=process.pid,time_unix=time.time()));code=process.wait()
    put(folder/'exit.json',dict(actual_returncode=code,time_unix=time.time()));return code
if __name__=='__main__':raise SystemExit(watch(sys.argv[2]) if sys.argv[1]=='--watch' else main(sys.argv[1]))
