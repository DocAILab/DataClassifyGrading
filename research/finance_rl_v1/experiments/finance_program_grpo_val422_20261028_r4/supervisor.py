"""One paired evaluation after audited parent exits, within the nightwatch cap."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from support import ROOT,TRAIN,load,native,put,verify,sha
from process_contract import is_training_writer

PY='/home/zhubh/finance_program_grpo_full_migrated_20261027_r3/env/bin/python'


def parent_ready():
    request=json.loads((ROOT/'candidate-request.json').read_bytes());proof=json.loads((ROOT/'reviews/parent-acceptance.json').read_bytes())
    parent=ROOT.parent/request['training_root']
    assert proof['independent_checkpoint_verified'] and proof['checkpoint_state_sha256']==request['checkpoint_sha256']
    last=sorted((parent/'attempts').glob('*'))[-1]
    for rel in ('supervision/worker-exit.json','launch/exit.json'):
        assert json.loads((last/rel).read_bytes())['actual_returncode']==0
    complete=json.loads((parent/'runs/completion.json').read_bytes())
    assert complete['completed'] and complete['episodes']==192 and 1<=complete['actual_optimizer_updates']<=6
    assert sha(parent/request['checkpoint_dir']/'state.pt')==request['checkpoint_sha256']
    assert sha(parent/'original.tar.gz')==proof['parent_archive_sha256']
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            cmd=(p/'cmdline').read_bytes();state=(p/'stat').read_text().split(') ',1)[1].split()[0]
            if is_training_writer(cmd,state,str(parent),str(ROOT)):raise ValueError('training writer still live')
        except (FileNotFoundError,ProcessLookupError):pass
    return proof


def main():
    verify();preflight=json.loads((ROOT/'reviews/preflight.json').read_bytes())
    if not preflight['passed'] or preflight['targets']!=422 or preflight['model_calls']!=0:raise ValueError('actual422 CPU preflight required')
    parent_ready();policy=json.loads((ROOT/'campaign-policy.json').read_bytes());budget=json.loads((ROOT/'evaluation-budget.json').read_bytes())
    if budget['maximum_calls']+budget['previous_calls']!=17724 or abs(budget['maximum_seconds']+budget['previous_GPU_seconds']-28800)>0.001:raise ValueError('immutable evaluation budget differs')
    run=ROOT/'supervision';run.mkdir(exist_ok=False);handles=[]
    lock=(ROOT/'writer.lock').open('a+');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    put(run/'started.json',dict(pid=os.getpid(),time_unix=time.time(),kind='GRPO_only_Val422'))
    try:
        native();sys.path.append(str(TRAIN))
        shared=load('paired_original_shared_locks',TRAIN/'supervisor.py');profile=load('paired_current_resource_profile',ROOT/'resource_runtime.py')
        from runtime_safety import guard_resources
        guard_resources(ROOT,preload=True);admitted=None
        for combo in profile.candidates(shared.snapshot(),policy):
            handles=shared.lock_combo(combo)
            if handles is None:handles=[];continue
            try:admitted=profile.recheck(shared.snapshot(),combo,policy)
            except ValueError:
                for h in handles:h.close()
                handles=[];continue
            break
        if admitted is None:raise RuntimeError('authorized GPUs/29.5GiB pre-admission/shared locks unavailable; no GPU started')
        parent_ready();now=time.time();deadline=min(now+budget['maximum_seconds'],budget['absolute_end_unix'])
        if deadline-now<60:raise TimeoutError('evaluation deadline exhausted')
        stage=dict(gpu_deadline_unix=deadline,model_call_limit=budget['maximum_calls'],evaluation_only=True)
        admission=dict(json.loads((TRAIN/'admission.json').read_bytes()),**admitted)
        admission.update(stage,gpu_started_unix=now,scientific_updates_allowed=0,maximum_optimizer_updates=0,
            source_denominator=422,expected_episodes=422,N_only=True,campaign_policy_sha256=sha(ROOT/'campaign-policy.json'))
        put(ROOT/'admission.json',admission)
        remaining=int(admission['gpu_deadline_unix']-time.time())-20
        if remaining<=20:raise TimeoutError('remaining nightwatch budget insufficient')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(admission['gpu_uuids']),PYTHONUTF8='1',TOKENIZERS_PARALLELISM='false')
        command=['/usr/bin/timeout','--signal=TERM','--kill-after=20s',str(remaining)+'s',PY,'-B',str(ROOT/'runner.py')]
        with (run/'worker.stdout.log').open('xb') as out,(run/'worker.stderr.log').open('xb') as err:
            p=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True,pass_fds=tuple(h.fileno() for h in handles+[lock]))
            put(run/'worker-start.json',dict(pid=p.pid,command=command,time_unix=time.time(),inherited_locks=4));code=p.wait()
        put(run/'worker-exit.json',dict(actual_returncode=code,time_unix=time.time()))
        if code:raise RuntimeError('actual paired worker exit '+str(code)+'; preserve originals')
        verify();assert len(json.loads((ROOT/'runs/completion.json').read_bytes())['rows'])==422
        put(run/'completion.json',dict(actual_returncode=0,time_unix=time.time()));return 0
    except BaseException as exc:
        put(run/'failure.json',dict(type=type(exc).__name__,error=str(exc),time_unix=time.time()));raise
    finally:
        for h in handles:h.close()
        lock.close()


def watch():
    folder=ROOT/'launch';folder.mkdir(exist_ok=False)
    with (folder/'stdout.log').open('xb') as out,(folder/'stderr.log').open('xb') as err:
        p=subprocess.Popen([PY,'-B',str(ROOT/'supervisor.py')],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err)
        put(folder/'started.json',dict(watcher_pid=os.getpid(),supervisor_pid=p.pid,time_unix=time.time()));code=p.wait()
    put(folder/'exit.json',dict(actual_returncode=code,time_unix=time.time()));return code


if __name__=='__main__':raise SystemExit(watch() if '--watch' in sys.argv else main())
