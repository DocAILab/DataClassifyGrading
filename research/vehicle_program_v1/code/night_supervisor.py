"""Independent task host: sleep inhibition, owned worker, bounded wall time, final audit."""
import argparse
from contextlib import contextmanager
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from artifacts import put, read, sha
from night_runner import validate_plan
from runner import exclusive_run
from api_backend import load_config
from night_resume import validate_resume_ledger


@contextmanager
def keep_awake(enabled=True):
    """Keep the system awake only while running; allow the display to turn off."""
    if enabled and os.name == 'nt':
        fn = ctypes.windll.kernel32.SetThreadExecutionState
        fn.argtypes = [ctypes.c_uint32]
        fn.restype = ctypes.c_uint32
        if not fn(0x80000001):
            raise RuntimeError('sleep_inhibition_failed')
        try:
            yield
        finally:
            fn(0x80000000)
    else:
        yield


def on_ac_power():
    if os.name != 'nt':
        return True
    class Power(ctypes.Structure):
        _fields_ = [('ac', ctypes.c_ubyte), ('flags', ctypes.c_ubyte), ('percent', ctypes.c_ubyte),
                    ('reserved', ctypes.c_ubyte), ('remaining', ctypes.c_ulong), ('full', ctypes.c_ulong)]
    value = Power()
    if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(value)):
        raise RuntimeError('power_status_unavailable')
    return value.ac == 1


@contextmanager
def owned_worker(command, *, cwd, stdout, stderr, flags, stop_file, grace):
    process = subprocess.Popen(command, cwd=cwd, stdout=stdout, stderr=stderr,
                               stdin=subprocess.DEVNULL, creationflags=flags)
    try:
        yield process
    finally:
        if process.poll() is None:
            Path(stop_file).touch(exist_ok=True)
            try:
                process.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                process.terminate(); process.wait(timeout=30)


def supervise(plan_path, *, resume=False):
    plan_path = Path(plan_path).resolve()
    plan = read(plan_path)
    study, manifest, seed = validate_plan(plan)
    config = load_config(Path(plan['package']) / 'night-api-profile.json')
    original_start = validate_resume_ledger(plan, study, seed, config) if resume else time.time()
    deadline = original_start + min(plan['supervisor_wall_seconds'], config['limits']['max_wall_seconds'])
    if time.time() >= deadline:
        raise RuntimeError('wall_budget_exhausted')
    home = Path(plan['home']); home.mkdir(parents=True, exist_ok=True)
    if plan['mode'] == 'api' and not on_ac_power():
        raise RuntimeError('ac_power_required')
    if Path(plan['stop_file']).exists():
        raise RuntimeError('stop_marker_present')
    if plan['mode'] == 'api':
        from api_discover import discover
        report = discover(home / ('connectivity-' + str(time.time_ns()) + '.json'))
        if not report.get('exact_model_listed') or report.get('http_status') != 200:
            raise RuntimeError('provider_preflight_failed')
    (home / 'supervisor-lock').mkdir(exist_ok=True)
    with exclusive_run(home / 'supervisor-lock'), keep_awake(plan['mode'] == 'api'):
        import psutil
        me = psutil.Process()
        started = time.time()
        meta = {'state': 'starting', 'supervisor_pid': os.getpid(), 'supervisor_created_unix': me.create_time(),
                'plan_sha256': sha(plan_path), 'started_unix': started, 'wall_deadline_unix': deadline,
                'system_sleep_inhibited': plan['mode'] == 'api', 'display_kept_on': False}
        put(home / 'supervisor.json', meta, replace=True)
        command = [sys.executable.replace('pythonw.exe', 'python.exe'), '-u',
                   str(Path(__file__).with_name('night_runner.py')), '--plan', str(plan_path)]
        if resume:
            command.append('--resume')
        flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        timed_out = False
        with (home / 'worker.stdout.log').open('ab') as stdout, (home / 'worker.stderr.log').open('ab') as stderr, \
                owned_worker(command, cwd=Path(__file__).parent, stdout=stdout, stderr=stderr,
                             flags=flags, stop_file=plan['stop_file'],
                             grace=plan['policy']['shutdown_grace_seconds']) as process:
            meta.update(state='running', worker_pid=process.pid,
                        worker_created_unix=psutil.Process(process.pid).create_time())
            put(home / 'supervisor.json', meta, replace=True)
            while process.poll() is None:
                if time.time() >= deadline:
                    timed_out = True
                    stop = Path(plan['stop_file']); stop.parent.mkdir(parents=True, exist_ok=True)
                    stop.touch(exist_ok=True)
                    try:
                        process.wait(timeout=plan['policy']['shutdown_grace_seconds'])
                    except subprocess.TimeoutExpired:
                        # This is the Popen handle for our own child, never an arbitrary PID.
                        process.terminate(); process.wait(timeout=30)
                    break
                time.sleep(1.)
        meta.update(state='stopped' if timed_out or process.returncode else 'complete',
                    worker_exit_code=process.returncode, finished_unix=time.time(),
                    supervisor_wall_stop=timed_out, system_sleep_inhibited=False)
        # Audits run only after generation exits. gold is never read by the worker.
        if (Path(plan['output']) / 'run.json').exists():
            try:
                from replay import replay
                from score import score_run
                put(home / 'replay.json', replay(plan['package'], plan['output']), replace=True)
                put(home / 'score.json', score_run(plan['package'], plan['output']), replace=True)
                meta['post_run_audit'] = 'pass'
            except Exception as exc:
                meta['post_run_audit'] = type(exc).__name__
        put(home / 'supervisor.json', meta, replace=True)
        return meta


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    try:
        result = supervise(args.plan, resume=args.resume)
        raise SystemExit(0 if result['state'] == 'complete' and result.get('post_run_audit') == 'pass' else 2)
    except Exception as exc:
        # Never write exception text or HTTP bodies into a launcher log.
        home = Path(read(args.plan)['home'])
        put(home / 'supervisor-failure.json', {'error_type': type(exc).__name__, 'time_unix': time.time()}, replace=True)
        raise SystemExit(3)
