"""One detached diagnostic worker, original shared locks and absolute deadline."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
E2 = ROOT.parent / 'finance_program_student_b80_eval_20261021_r2'
sys.path.insert(0, str(E2))
from launch_training import snapshot, _lock_combo, put, PYTHON, DEPS
from resource_gate import candidates, recheck

DEADLINE = 1792575276.1708825


def verify():
    lock = json.loads((ROOT / 'freeze.json').read_bytes())
    for entry in lock['files']:
        path = ROOT / entry['path']
        if path.is_symlink() or not path.is_file() or path.stat().st_size != entry['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError('new smoke freeze changed: '+entry['path'])
    old = json.loads((E2 / 'reviews/evaluation-freeze-01.json').read_bytes())
    for entry in old['files']:
        if hashlib.sha256((E2 / entry['path']).read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError('frozen E2 dependency changed: '+entry['path'])
    for pid in (2731610, 2731611, 2731641, 2731642):
        stat = Path('/proc') / str(pid) / 'stat'
        if stat.exists() and stat.read_text().split(') ', 1)[1].split()[0] != 'Z':
            raise RuntimeError('old evaluation tree remains live')


def main():
    folder = ROOT / 'supervision'
    folder.mkdir(exist_ok=False)
    put(folder / 'started.json', dict(pid=os.getpid(), time_unix=time.time(), deadline=DEADLINE))
    handles = []
    try:
        verify()
        admission = None
        while time.time() < DEADLINE-20:
            for combo in candidates(snapshot(), max_sequence_tokens=32768):
                if len(combo['gpu_indices']) != 3:
                    continue
                handles = _lock_combo(combo)
                if handles is None:
                    handles = []; continue
                try:
                    admission = recheck(snapshot(), combo)
                except ValueError:
                    for h in handles: h.close()
                    handles = []; continue
                break
            if admission is not None:
                break
            time.sleep(10)
        if admission is None:
            raise TimeoutError('no safe three-card combination before unchanged deadline')
        original = json.loads((E2 / 'admission-evaluation-01.json').read_bytes())
        original.update(admission)
        original['gpu_deadline_unix'] = DEADLINE
        put(ROOT / 'admission.json', original)
        verify()
        cmd = ['/usr/bin/timeout', '--signal=TERM', '--kill-after=20s', str(int(DEADLINE-time.time())-20)+'s',
               PYTHON, '-B', str(ROOT / 'run_smoke.py')]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(original['gpu_uuids']), PYTHONPATH=DEPS,
                   PYTHONUTF8='1', TOKENIZERS_PARALLELISM='false')
        with (folder / 'stdout.log').open('xb') as out, (folder / 'stderr.log').open('xb') as err:
            worker = subprocess.Popen(cmd, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                      start_new_session=True, pass_fds=tuple(h.fileno() for h in handles))
            put(folder / 'worker-start.json', dict(pid=worker.pid, command=cmd, time_unix=time.time()))
            code = worker.wait()
        put(folder / 'worker-exit.json', dict(actual_returncode=code, time_unix=time.time(),
                                               completion_exists=(ROOT / 'runs/completion.json').is_file()))
        return code
    except BaseException as exc:
        put(folder / 'failure.json', dict(error_type=type(exc).__name__, error=str(exc), time_unix=time.time()))
        raise
    finally:
        for h in handles: h.close()


def watch():
    folder = ROOT / 'launch'
    folder.mkdir(exist_ok=False)
    with (folder / 'stdout.log').open('xb') as out, (folder / 'stderr.log').open('xb') as err:
        p = subprocess.Popen([PYTHON, '-B', str(ROOT / 'launch_smoke.py')], cwd=ROOT,
            stdin=subprocess.DEVNULL, stdout=out, stderr=err)
        put(folder / 'started.json', dict(watcher_pid=os.getpid(), supervisor_pid=p.pid, time_unix=time.time()))
        code = p.wait()
    put(folder / 'exit.json', dict(actual_returncode=code, time_unix=time.time()))
    return code


if __name__ == '__main__':
    raise SystemExit(watch() if sys.argv[1:] == ['--watch'] else main())
