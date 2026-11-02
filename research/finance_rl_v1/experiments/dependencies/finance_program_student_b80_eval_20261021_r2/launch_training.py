"""One detached B80 training worker, shared locks and a non-resetting hard limit.

No model import, device allocation, arbitrary task command, resume or retry.
The independently prepared evaluation later uses this SAME absolute deadline.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from resource_gate import candidates, recheck, remaining_seconds

ROOT = Path(__file__).resolve().parent
PYTHON = '/home/zhubh/finance_program_grpo_full_migrated_20261027_r3/dependencies/finance_program_5090_smoke_20261010/env/bin/python'
BASE = '/home/zhubh/finance_program_grpo_full_migrated_20261027_r3/dependencies/finance_program_5090_smoke_20261010/models/Qwen3.5-9B'
ORIGINAL = '/home/zhubh/finance_program_grpo_full_migrated_20261027_r3/dependencies/finance_program_student_sft_20261016_r1/formal/final'
LOCK_ROOT = Path('/home/zhubh/finance_program_grpo_full_migrated_20261027_r3/dependencies/finance_program_5090_smoke_20261010')
DEPS = '/home/zhubh/finance_program_grpo_full_migrated_20261027_r3/dependencies/finance_program_adapter_compare_20261012_r2/deps'


def put(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def snapshot():
    def query(args):
        return subprocess.check_output(['nvidia-smi'] + args + ['--format=csv,noheader,nounits'], text=True, timeout=30)
    devices = query(['--query-gpu=index,uuid,memory.total,memory.free,compute_cap'])
    processes = query(['--query-compute-apps=gpu_uuid,pid,used_memory'])
    pids = {}
    for line in processes.splitlines():
        parts = [p.strip() for p in line.split(',')]
        if len(parts) != 3:
            raise ValueError('unparseable actual compute process')
        pids.setdefault(parts[0], []).append(int(parts[1]))
    rows = []
    for line in devices.splitlines():
        index, uuid, total, free, capability = [p.strip() for p in line.split(',')]
        rows.append(dict(index=int(index), uuid=uuid, total_bytes=int(total)*1024**2,
                         free_bytes=int(free)*1024**2, bf16_compatible=int(capability.split('.')[0]) >= 8,
                         other_compute_pids=pids.get(uuid, [])))
    return rows


def _lock_combo(combo):
    import fcntl
    handles = []
    try:
        for index in combo['gpu_indices']:
            handle = (LOCK_ROOT / f'gpu{index}-experiment.lock').open('a+')
            handles.append(handle)
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return handles
    except BlockingIOError:
        for handle in handles: handle.close()
        return None
    except BaseException:
        for handle in handles: handle.close()
        raise


def main():
    if len(sys.argv) != 1:
        raise ValueError('no supervisor command or budget overrides permitted')
    audit_path = ROOT / 'preflight-cpu-01/tokenizer-audit.json'
    audit = json.loads(audit_path.read_bytes())
    from training_contract import TRAIN_SHA256, MANIFEST_SHA256
    if (audit['rows'] != 80 or audit['train_sha256'] != TRAIN_SHA256
            or audit['manifest_sha256'] != MANIFEST_SHA256 or audit['max_input_tokens'] > 32768
            or len(audit['per_source']) != 80 or any(r['truncated'] for r in audit['per_source'])):
        raise ValueError('actual original CPU tokenizer80 proof missing or changed')
    run = ROOT / 'supervision-training-01'
    run.mkdir(exist_ok=False)
    put(run/'supervisor-start.json', dict(pid=os.getpid(), time_unix=time.time(),
                                         command=sys.argv, gpu_started=False))
    handles = []
    worker = None
    try:
        wait_deadline = time.monotonic() + 86400
        while time.monotonic() < wait_deadline:
            observed = snapshot()
            admitted = None
            for combo in candidates(observed, max_sequence_tokens=audit['max_input_tokens']):
                handles = _lock_combo(combo)
                if handles is None:
                    handles = []
                    continue
                try:
                    admitted = recheck(snapshot(), combo)
                except ValueError:
                    for handle in handles: handle.close()
                    handles = []
                    continue
                break
            if admitted is not None:
                break
            time.sleep(30)
        else:
            raise TimeoutError('24h bounded resource wait exhausted, no GPU worker started')
        started = time.time()
        admission = dict(admitted, base_model_dir=BASE, original_adapter_dir=ORIGINAL,
                         gpu_started_unix=started, gpu_deadline_unix=started+21600,
                         token_audit_sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest())
        put(ROOT/'admission-training-01.json', admission)
        timeout = shutil.which('timeout')
        if not timeout:
            raise RuntimeError('GNU hard-deadline timeout executable unavailable')
        duration = int(remaining_seconds(admission)) - 20
        if duration <= 0:
            raise TimeoutError('no hard-deadline time left before worker launch')
        command = [timeout, '--signal=TERM', '--kill-after=20s', str(duration)+'s',
                   PYTHON, '-B', str(ROOT/'train_student.py'), '--root', str(ROOT),
                   '--admission', str(ROOT/'admission-training-01.json')]
        env = dict(os.environ)
        env.update(CUDA_VISIBLE_DEVICES=','.join(admission['gpu_uuids']), PYTHONPATH=DEPS,
                   PYTHONUTF8='1', TOKENIZERS_PARALLELISM='false')
        with (run/'worker.stdout.log').open('xb') as out, (run/'worker.stderr.log').open('xb') as err:
            worker = subprocess.Popen(command, cwd=ROOT, env=env, stdout=out, stderr=err,
                                      start_new_session=True, pass_fds=tuple(h.fileno() for h in handles))
            put(run/'worker-start.json', dict(pid=worker.pid, process_group=worker.pid, command=command,
                                              time_unix=time.time(), inherited_lock_count=len(handles)))
            code = worker.wait()
        put(run/'worker-exit.json', dict(pid=worker.pid, actual_returncode=code, time_unix=time.time(),
                                         completion_exists=(ROOT/'runs/training/completion.json').is_file()))
        if code != 0 or not (ROOT/'runs/training/completion.json').is_file():
            raise RuntimeError(f'training worker failed with real return code {code}; no automatic retry')
        put(run/'supervisor-exit.json', dict(actual_returncode=0, time_unix=time.time(),
                                            evaluation_complete=False, remaining_seconds=max(0, admission['gpu_deadline_unix']-time.time())))
        return 0
    except BaseException as exc:
        put(run/'failure.json', dict(error_type=type(exc).__name__, error=str(exc), time_unix=time.time(),
                                     worker_pid=worker.pid if worker else None, restart_allowed=False))
        # Inherited lock descriptors plus GNU timeout protect a still-live
        # worker if this supervisor unexpectedly dies; no external process is killed.
        raise
    finally:
        for handle in handles: handle.close()


def watch():
    """Detached outer waiter records the supervisor's real OS return code."""
    folder = ROOT/'launch-watch-01'
    folder.mkdir(exist_ok=False)
    with (folder/'supervisor.stdout.log').open('xb') as out, (folder/'supervisor.stderr.log').open('xb') as err:
        process = subprocess.Popen([PYTHON, '-B', str(ROOT/'launch_training.py')],
                                   cwd=ROOT, stdout=out, stderr=err)
        put(folder/'started.json', dict(watcher_pid=os.getpid(), supervisor_pid=process.pid,
                                        time_unix=time.time()))
        code = process.wait()
    put(folder/'supervisor-os-exit.json', dict(pid=process.pid, actual_returncode=code,
                                              time_unix=time.time()))
    return code


if __name__ == '__main__':
    raise SystemExit(watch() if sys.argv[1:] == ['--watch'] else main())
