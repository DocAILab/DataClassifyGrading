"""One writer and one bounded inference process with inherited shared GPU locks."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from support import ROOT, N100, TRAIN, native, load, put, verify

native()
sys.path.append(str(TRAIN))
from resource_profile import candidates, recheck, PINNED_UUIDS
from runtime_safety import guard_resources
old = load('termination_first_shared_resource_supervisor',TRAIN/'supervisor.py')
PY = '/home/zhubh/finance_program_grpo_full_migrated_20261027_r3/dependencies/finance_program_5090_smoke_20261010/env/bin/python'


def main():
    verify()
    proof = json.loads((ROOT/'reviews/preflight.json').read_bytes())
    assert proof['passed'] and proof['exact_initial_targets']==100 and proof['new_model_calls']==0
    run = ROOT/'supervision'
    run.mkdir(exist_ok=False)
    handles = []
    lock = (ROOT/'writer.lock').open('a+')
    fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    put(run/'started.json',dict(pid=os.getpid(),time_unix=time.time()))
    try:
        guard_resources(ROOT,preload=True)
        admitted = None
        for combo in candidates(old.snapshot()):
            if combo['gpu_indices'] != [0,2,3] or combo['gpu_uuids'] != [PINNED_UUIDS[i] for i in (0,2,3)]:
                continue
            handles = old.lock_combo(combo)
            if handles is None:
                handles=[]
                continue
            try:
                admitted = recheck(old.snapshot(),combo)
            except ValueError:
                for h in handles:
                    h.close()
                handles=[]
                continue
            break
        if admitted is None:
            raise RuntimeError('shared 0/2/3 GPU locks or 30GiB free admission unavailable')
        historical = json.loads((TRAIN/'admission.json').read_bytes())
        started = time.time()
        admission = dict(historical,**admitted)
        admission.update(gpu_started_unix=started,gpu_deadline_unix=started+8*3600,
                         evaluation_only=True,evaluation_root=ROOT.name,only_student_arm=True,
                         model_call_limit=4200,source_denominator=100,failed_first=42,
                         new_window_reason='user_authorized_20261025_termination_repair_test')
        put(ROOT/'admission.json',admission)
        env = dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(admission['gpu_uuids']),
                   PYTHONUTF8='1',TOKENIZERS_PARALLELISM='false')
        command = ['/usr/bin/timeout','--signal=TERM','--kill-after=20s',str(8*3600-20)+'s',PY,'-B',str(ROOT/'runner.py')]
        with (run/'worker.stdout.log').open('xb') as out,(run/'worker.stderr.log').open('xb') as err:
            proc = subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                                    stdout=out,stderr=err,start_new_session=True,
                                    pass_fds=tuple(h.fileno() for h in handles+[lock]))
            put(run/'worker-start.json',dict(pid=proc.pid,command=command,time_unix=time.time(),inherited_locks=4))
            code = proc.wait()
        put(run/'worker-exit.json',dict(actual_returncode=code,time_unix=time.time()))
        if code != 0:
            raise RuntimeError('new SFT evaluation actual exit '+str(code)+'; no automatic retry')
        verify()
        complete = json.loads((ROOT/'runs/completion.json').read_bytes())
        assert len(complete['rows'])==100 and complete['N_only'] and complete['reused_results']==0
        put(run/'completion.json',dict(actual_returncode=0,time_unix=time.time()))
        return 0
    except BaseException as exc:
        put(run/'failure.json',dict(error_type=type(exc).__name__,error=str(exc),time_unix=time.time()))
        raise
    finally:
        for h in handles:
            h.close()
        lock.close()


def watch():
    folder = ROOT/'launch'
    folder.mkdir(exist_ok=False)
    with (folder/'stdout.log').open('xb') as out,(folder/'stderr.log').open('xb') as err:
        proc = subprocess.Popen([PY,'-B',str(ROOT/'supervisor.py')],cwd=ROOT,stdout=out,stderr=err)
        put(folder/'started.json',dict(watcher_pid=os.getpid(),supervisor_pid=proc.pid,time_unix=time.time()))
        code = proc.wait()
    put(folder/'exit.json',dict(actual_returncode=code,time_unix=time.time()))
    return code


if __name__=='__main__':
    raise SystemExit(watch() if '--watch' in sys.argv else main())
