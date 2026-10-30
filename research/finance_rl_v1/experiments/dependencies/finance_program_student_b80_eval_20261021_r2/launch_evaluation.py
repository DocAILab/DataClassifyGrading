"""Wait for the unique successful r3, then one paired run under original deadline."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from launch_training import snapshot, _lock_combo, put, PYTHON, BASE, ORIGINAL, DEPS
from resource_gate import candidates, recheck, remaining_seconds

ROOT = Path(__file__).resolve().parent
TRAINING = ROOT.parent / 'finance_program_student_b80_20261021_r3'
DEADLINE = 1792575276.1708825
LOADER_SHA = 'c7ecfba0b1866e5730b970200dbfc0adb441722bd85b66f57f44898305a4e990'


def verify_files():
    manifest = json.loads((ROOT/'reviews/evaluation-freeze-01.json').read_bytes())
    for entry in manifest['files']:
        path = ROOT / entry['path']
        if (path.is_symlink() or not path.is_file() or path.stat().st_size != entry['bytes']
                or hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']):
            raise ValueError('evaluation freeze changed: '+entry['path'])
    if hashlib.sha256((TRAINING/'train_student.py').read_bytes()).hexdigest() != LOADER_SHA:
        raise ValueError('admitted r3 loader changed')


def parent_ready():
    failure = TRAINING/'supervision-training-01/failure.json'
    if failure.exists():
        raise RuntimeError('r3 training failed; no paired GPU launch')
    checks = [TRAINING/'supervision-training-01/worker-exit.json',
              TRAINING/'launch-watch-01/supervisor-os-exit.json']
    if not all(p.is_file() for p in checks):
        return False
    if any(json.loads(p.read_bytes())['actual_returncode'] != 0 for p in checks):
        raise RuntimeError('r3 actual worker/supervisor return code is nonzero')
    # Parent completion is not enough: do not allocate while a live writer remains.
    for name, keys in [('launch-watch-01/started.json', ('watcher_pid','supervisor_pid')),
                       ('supervision-training-01/worker-start.json', ('pid',))]:
        record = json.loads((TRAINING/name).read_bytes())
        for key in keys:
            path = Path('/proc') / str(record[key]) / 'stat'
            if path.exists() and path.read_text().split(') ',1)[1].split()[0] != 'Z':
                return False
    complete = json.loads((TRAINING/'runs/training/completion.json').read_bytes())
    if (complete['cpu_test_only'] is not False or complete['rows'] != 80
            or len(complete['optimizer_steps']) != 10
            or complete['admission']['gpu_deadline_unix'] != DEADLINE):
        raise ValueError('r3 actual 80/10/original deadline completion mismatch')
    return True


def main():
    if len(sys.argv) != 1:
        raise ValueError('no budget/task overrides')
    folder = ROOT/'supervision-evaluation-01'
    folder.mkdir(exist_ok=False)
    put(folder/'supervisor-start.json', dict(pid=os.getpid(), time_unix=time.time(),
                                            state='waiting_unique_r3', gpu_deadline_unix=DEADLINE))
    handles=[]
    try:
        verify_files()
        while not parent_ready():
            if time.time() >= DEADLINE:
                raise TimeoutError('original GPU deadline ended waiting r3; not extended')
            time.sleep(5)
        verify_files()
        put(folder/'parent-accepted.json', dict(time_unix=time.time(), training_root=str(TRAINING),
            completion_sha256=hashlib.sha256((TRAINING/'runs/training/completion.json').read_bytes()).hexdigest()))
        admitted=None
        while time.time() < DEADLINE-20:
            for combo in candidates(snapshot(), max_sequence_tokens=32768):
                if len(combo['gpu_indices']) != 3:
                    continue
                handles = _lock_combo(combo)
                if handles is None:
                    handles=[]
                    continue
                try:
                    admitted = recheck(snapshot(), combo)
                except ValueError:
                    for handle in handles:handle.close()
                    handles=[]
                    continue
                break
            if admitted is not None:break
            time.sleep(10)
        if admitted is None:
            raise TimeoutError('no three-card headroom before original deadline')
        prior=json.loads((TRAINING/'admission-training-01.json').read_bytes())
        admission=dict(admitted, base_model_dir=BASE, original_adapter_dir=ORIGINAL,
            gpu_started_unix=prior['gpu_started_unix'], gpu_deadline_unix=DEADLINE,
            training_root=str(TRAINING), child_adapter_dir=str(TRAINING/'runs/training/final-adapter'),
            runtime_loader_sha256=LOADER_SHA)
        put(ROOT/'admission-evaluation-01.json', admission)
        timeout=shutil.which('timeout')
        if timeout is None:raise RuntimeError('hard timeout unavailable')
        duration=int(remaining_seconds(admission))-20
        if duration<=0:raise TimeoutError('no original budget remains')
        command=[timeout,'--signal=TERM','--kill-after=20s',str(duration)+'s',PYTHON,'-B',
                 str(ROOT/'paired_eval.py'),'--root',str(ROOT),'--admission',str(ROOT/'admission-evaluation-01.json')]
        env=dict(os.environ)
        env.update(CUDA_VISIBLE_DEVICES=','.join(admission['gpu_uuids']),PYTHONPATH=DEPS,
                   PYTHONUTF8='1',TOKENIZERS_PARALLELISM='false')
        with (folder/'worker.stdout.log').open('xb') as out,(folder/'worker.stderr.log').open('xb') as err:
            worker=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                stdout=out,stderr=err,start_new_session=True,pass_fds=tuple(h.fileno() for h in handles))
            put(folder/'worker-start.json',dict(pid=worker.pid,time_unix=time.time(),command=command,
                                               inherited_lock_count=len(handles)))
            code=worker.wait()
        put(folder/'worker-exit.json',dict(pid=worker.pid,actual_returncode=code,time_unix=time.time(),
            completion_exists=(ROOT/'runs/evaluation/completion.json').is_file()))
        if code or not (ROOT/'runs/evaluation/completion.json').is_file():
            raise RuntimeError('paired worker real exit '+str(code)+'; no automatic retry')
        put(folder/'supervisor-exit.json',dict(actual_returncode=0,time_unix=time.time(),final_delivery_complete=False))
        return 0
    except BaseException as exc:
        put(folder/'failure.json',dict(error_type=type(exc).__name__,error=str(exc),time_unix=time.time()))
        raise
    finally:
        for handle in handles:handle.close()


def watch():
    folder=ROOT/'launch-evaluation-watch-01'
    folder.mkdir(exist_ok=False)
    with (folder/'supervisor.stdout.log').open('xb') as out,(folder/'supervisor.stderr.log').open('xb') as err:
        process=subprocess.Popen([PYTHON,'-B',str(ROOT/'launch_evaluation.py')],cwd=ROOT,
                                 stdout=out,stderr=err,stdin=subprocess.DEVNULL)
        put(folder/'started.json',dict(watcher_pid=os.getpid(),supervisor_pid=process.pid,time_unix=time.time()))
        code=process.wait()
    put(folder/'supervisor-os-exit.json',dict(pid=process.pid,actual_returncode=code,time_unix=time.time()))
    return code


if __name__=='__main__':raise SystemExit(watch() if sys.argv[1:]==['--watch'] else main())
