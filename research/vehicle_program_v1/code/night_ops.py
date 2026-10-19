"""Offline preflight, compact progress, graceful stop and explicit resume checks."""
import argparse
import json
from pathlib import Path
import time

from api_backend import CREDENTIAL_FILE, ReferenceCounter, load_config
from artifacts import read
from night_runner import public_status, validate_plan, TRANSIENT
from night_supervisor import on_ac_power
from night_resume import validate_resume_ledger


def supervisor_alive(plan):
    import psutil
    p = Path(plan['home']) / 'supervisor.json'
    if not p.exists():
        return False
    meta = read(p)
    try:
        process = psutil.Process(meta['supervisor_pid'])
        return (abs(process.create_time() - meta['supervisor_created_unix']) < .01
                and str(Path(__file__).with_name('night_supervisor.py')) in process.cmdline())
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False


def preflight(plan, *, resume=False):
    study, manifest, seed = validate_plan(plan)
    if supervisor_alive(plan):
        raise RuntimeError('night_already_running')
    output = Path(plan['output'])
    if output.exists() and not resume:
        raise RuntimeError('existing_run_use_explicit_resume')
    if resume:
        start = validate_resume_ledger(plan, study, seed, load_config(Path(plan['package']) / 'night-api-profile.json'))
        wall = min(plan['supervisor_wall_seconds'], read(Path(plan['package']) / 'night-api-profile.json')['limits']['max_wall_seconds'])
        if time.time() >= start + wall:
            raise RuntimeError('wall_budget_exhausted')
    if plan['mode'] == 'api':
        if not CREDENTIAL_FILE.is_file():
            raise RuntimeError('credential_source_missing')
        if not on_ac_power():
            raise RuntimeError('ac_power_required')
        ReferenceCounter(read(Path(plan['package']) / 'night-api-profile.json')['tokenizer_path'])
    return {'ready': True, 'mode': plan['mode'], 'expected_sources': len(study['rows']), 'preserved_sources': len(seed),
            'pending_sources_before_start': len(study['rows']) - len(seed), 'network_requests': 0,
            'credential_value_read': False, 'scheduled_trigger': None}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--action', choices=('check', 'status', 'stop', 'resume-check'), required=True)
    a = p.parse_args(); plan = read(a.plan)
    if a.action in ('check', 'resume-check'):
        result = preflight(plan, resume=a.action == 'resume-check')
        if a.action == 'resume-check':
            stop = Path(plan['stop_file'])
            if stop.exists():
                stop.rename(stop.with_name('STOP.' + str(time.time_ns())))
    elif a.action == 'stop':
        stop = Path(plan['stop_file']); stop.parent.mkdir(parents=True, exist_ok=True); stop.touch(exist_ok=True)
        result = {'stop_requested': True, 'force_kill': False, 'inflight_requests': 'allowed to finish or reach timeout'}
    else:
        result = public_status(plan['output'])
        result['supervisor_alive'] = supervisor_alive(plan)
        p = Path(plan['home']) / 'supervisor.json'
        if p.exists():
            result['supervisor'] = read(p)
        failure = Path(plan['home']) / 'supervisor-failure.json'
        if failure.exists():
            result['supervisor_failure'] = read(failure)
    print(json.dumps(result))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(json.dumps({'ready': False, 'error_type': type(exc).__name__,
                          'reason': str(exc) if isinstance(exc, RuntimeError) else 'validation_failed'}))
        raise SystemExit(2)
