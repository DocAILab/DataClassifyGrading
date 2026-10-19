"""Unattended, bounded evaluation; preserve old failures and never reroll requests."""
import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from copy import deepcopy
import json
from pathlib import Path
import shutil
import threading
import time

from api_backend import APIBackend, APIBudgetExceeded, APIError, ReferenceCounter, api_generation, api_identity, load_config, read_credential
from api_parallel import SharedBudget, source_receipt, validate_scheduler
from artifacts import BUDGET, fingerprint, load_controller, load_study, put, read, sha, require_private_output
from backends import ScriptedBackend
from runner import environment, exclusive_run, members, read_receipts, write_summary
from night_resume import TRANSIENT, effective_cooldown, validate_resume_ledger


class NightBudget(SharedBudget):
    def __init__(self, path, config, scheduler, counter, *, policy, stop_file, carry_cny=0., **kwargs):
        self.policy, self.stop_file, self.carry_cny = policy, Path(stop_file), carry_cny
        self.cooldown_until = 0.
        super().__init__(path, config, scheduler, counter, **kwargs)
        self.cooldown_until = effective_cooldown(self.state, policy)
        # Known failed requests stay charged and are not retried. Only fresh sources continue.
        self.stopped = any(a['state'] not in TRANSIENT | {'response_received'} for a in self.state['attempts'])

    def accounted_cny(self):
        return self.carry_cny + super().accounted_cny()

    def check(self, count=1):
        if self.stop_file.exists():
            raise APIBudgetExceeded('user_stop_requested')
        super().check(count)

    def reservation_delay(self, now):
        return max(0., self.cooldown_until - now)

    def finish(self, entry, outcome, usage=None):
        with self.lock:
            was_stopped = self.stopped
            if outcome in TRANSIENT and not was_stopped:
                delay = self.policy['rate_limit_cooldown_seconds'] if outcome == 'http_429' else self.policy['transient_cooldown_seconds']
                self.cooldown_until = max(self.cooldown_until, self.clock() + delay)
                self.state['cooldown_until_unix'] = self.cooldown_until
            super().finish(entry, outcome, usage)
            if outcome in TRANSIENT and not was_stopped:
                self.stopped = False


def validate_plan(plan):
    if plan.get('schema') != 'vehicle-nontd-night-v1' or plan.get('mode') not in ('api', 'scripted_test'):
        raise ValueError('night_plan_schema')
    package = Path(plan['package'])
    home = Path(plan['home']).resolve()
    require_private_output(home)
    if Path(plan['output']).resolve() != home / 'run' or Path(plan['stop_file']).resolve() != home / 'STOP':
        raise ValueError('night_output_or_stop_path_differs')
    study, manifest = load_study(package)
    for name, expected in plan['files'].items():
        if sha(package / name) != expected:
            raise ValueError('night_package_file_changed')
    if sha(package / 'manifest.json') != plan['package_manifest_sha256']:
        raise ValueError('night_package_manifest_changed')
    validate_scheduler(plan['scheduler'])
    provenance = read(package / 'seed-provenance.json')
    seed = read_receipts(package / 'seed', study['expected_ids'])
    if set(seed) != set(provenance['receipts']):
        raise ValueError('night_seed_members_changed')
    for n, ident in enumerate(study['expected_ids'], 1):
        if ident in seed and sha(package / f'seed/sources/{n:04d}/receipt.json') != provenance['receipts'][ident]['receipt_sha256']:
            raise ValueError('night_seed_receipt_changed')
    return study, manifest, seed


def public_status(output):
    """Small status file only; never prompts, response content, IDs or credentials."""
    p = Path(output) / 'night-status.json'
    return read(p) if p.exists() else {'state': 'not_started'}


def run_night(plan, *, resume=False, transport=None, counter=None, key=None):
    study, manifest, seed = validate_plan(plan)
    ids, output = study['expected_ids'], Path(plan['output'])
    config = load_config(Path(plan['package']) / 'night-api-profile.json')
    carry = read(Path(plan['package']) / 'seed-provenance.json')['usage_plus_uncertain_reserves_cny']
    kind = 'hosted_qwen35_9b_api' if plan['mode'] == 'api' else 'scripted_cpu_test'
    identity = {'schema': 4, 'arm': 'NONT-D', 'backend': kind, 'study_sha256': manifest['study_sha256'],
                'code_sha256': fingerprint(manifest['code']), 'expected_ids': ids, 'generation': api_generation(),
                'budget': BUDGET, 'environment': environment(), 'api_service': api_identity(config),
                'night_plan_sha256': fingerprint(plan), 'scheduler': plan['scheduler'],
                'seed_provenance_sha256': sha(Path(plan['package']) / 'seed-provenance.json'),
                'carried_usage_plus_uncertain_reserves_cny': carry,
                'comparison_scope': 'preserved pilot prefix plus night continuation; scheduler/service profiles disclosed separately'}
    if output.exists() and not resume:
        raise FileExistsError('night_output_exists')
    if resume and not (output / 'run.json').is_file():
        raise RuntimeError('resume_run_identity_missing')
    output.mkdir(parents=True, exist_ok=resume)
    with exclusive_run(output):
        if (output / 'run.json').exists():
            identity['wall_started_at_unix'] = validate_resume_ledger(plan, study, seed, config)
            if read(output / 'run.json') != identity:
                raise ValueError('night_run_identity_changed')
        else:
            if set(p.name for p in output.iterdir()) != {'.run.lock'}:
                raise ValueError('night_output_unidentified')
            identity['wall_started_at_unix'] = time.time()
            put(output / 'run.json', identity)
            for n, ident in enumerate(ids, 1):
                if ident in seed:
                    shutil.copytree(Path(plan['package']) / f'seed/sources/{n:04d}', output / f'sources/{n:04d}')
        receipts = read_receipts(output, ids)
        halted = None
        for n, ident in enumerate(ids, 1):
            folder = output / f'sources/{n:04d}'
            if ident not in receipts and folder.exists():
                # Preserve the partial evidence; manual review is required, never reroll it.
                r = {'id': ident, 'status': 'interrupted', 'terminated': False, 'stop_reason': 'interrupted_before_commit',
                     'seconds': None, 'model_calls': len(list((folder / 'calls').glob('*.json'))), 'files': members(folder)}
                put(folder / 'receipt.json', r); receipts[ident] = r
                halted = 'interrupted_source_requires_review'
        if any(r['status'] == 'interrupted' for r in receipts.values()):
            halted = 'interrupted_source_requires_review'
        remaining = [(row, output / f'sources/{n:04d}') for n, row in enumerate(study['rows'], 1) if row['id'] not in receipts]
        config_folder = output / 'backend'
        if remaining and not halted:
            counter = counter or (ReferenceCounter(config['tokenizer_path']) if plan['mode'] == 'api' else lambda messages: 100)
            key = key if key is not None else (read_credential() if plan['mode'] == 'api' else 'fictional-night-test-only')
        else:
            counter = counter or (lambda messages: 100)
        budget = NightBudget(config_folder / 'api-budget.json', config, plan['scheduler'], counter,
                             policy=plan['policy'], stop_file=plan['stop_file'], carry_cny=carry,
                             started_at_unix=identity['wall_started_at_unix'])
        if budget.stopped:
            halted = 'uncertain_or_fatal_attempt_requires_review'
        builder = load_controller()
        active = {}
        newly_finished, consecutive_errors = 0, 0
        started = time.monotonic()

        def status(state):
            with budget.lock:
                summary = write_summary(output, ids, receipts, kind, halted=halted)
                attempts = budget.state['attempts']
                value = {'state': state, 'updated_unix': time.time(), 'halted': halted,
                         'expected_sources': len(ids), 'preserved_sources': len(seed),
                         'committed_sources': len(receipts), 'pending_sources': len(ids) - len(receipts),
                         'newly_finished_sources': newly_finished, 'active_sources': len(active),
                         'source_errors': sum(r['status'] != 'completed' for r in receipts.values()),
                         'legal_terminations': sum(r['terminated'] for r in receipts.values()),
                         'new_http_attempts': len(attempts), 'cost_cap_cny': config['limits']['max_estimated_cny'],
                         'usage_plus_uncertain_reserves_cny': budget.accounted_cny(),
                         'uncertain_new_requests': sum(a['state'] != 'response_received' for a in attempts),
                         'carried_usage_plus_uncertain_reserves_cny': carry,
                         'wall_started_at_unix': identity['wall_started_at_unix'],
                         'wall_deadline_unix': identity['wall_started_at_unix'] + config['limits']['max_wall_seconds'],
                         'elapsed_this_process_seconds': time.monotonic() - started, 'invoice': False}
                put(output / 'night-status.json', value, replace=True)
            return value

        def work(row, folder):
            kwargs = {'transport': transport} if transport else {}
            if plan['mode'] == 'scripted_test' and transport is None:
                def scripted(url, secret, body, timeout):
                    if plan.get('test_delay_seconds'):
                        time.sleep(plan['test_delay_seconds'])
                    return {'model': config['model'], 'usage': {'prompt_tokens': 100, 'completion_tokens': 10},
                            'choices': [{'message': {'content': ScriptedBackend()(body['messages'])}, 'finish_reason': 'stop'}]}
                kwargs = {'transport': scripted}
            generator = APIBackend(config, config_folder, key=key, counter=counter, budget=budget, **kwargs)
            generator.before_source()
            return source_receipt(builder, study, row, folder, generator, kind=kind)

        status('running' if remaining and not halted else 'stopped')
        if remaining and not halted:
            pending = iter(remaining)
            with ThreadPoolExecutor(max_workers=plan['scheduler']['workers']) as pool:
                def submit():
                    nonlocal halted
                    if halted:
                        return
                    try:
                        budget.check(BUDGET['max_planner_attempts'] + BUDGET['max_skill_steps'])
                    except APIError as exc:
                        halted = str(exc); return
                    item = next(pending, None)
                    if item is not None:
                        active[pool.submit(work, *item)] = item[0]['id']
                for _ in range(min(plan['scheduler']['workers'], len(remaining))):
                    submit()
                while active:
                    done, _ = wait(active, timeout=plan['policy']['heartbeat_seconds'], return_when=FIRST_COMPLETED)
                    for future in done:
                        ident = active.pop(future)
                        try:
                            receipt = future.result(); receipts[ident] = receipt; newly_finished += 1
                            if receipt['status'] == 'runner_error':
                                halted = 'runner_error'
                            elif receipt['status'] == 'backend_error':
                                folder = output / f'sources/{ids.index(ident) + 1:04d}'
                                error_codes = [read(p)['generation'].get('error') for p in (folder / 'calls').glob('*.json')
                                               if read(p)['error_type'] is not None]
                                controls = {'user_stop_requested', 'wall_budget_exhausted',
                                            'estimated_cost_budget_exhausted', 'request_budget_exhausted'}
                                control_errors = [code for code in error_codes if code in controls]
                                if control_errors:
                                    halted = control_errors[0]
                                elif any(code not in TRANSIENT for code in error_codes):
                                    halted = 'fatal_backend_error'
                                consecutive_errors += 1
                                if consecutive_errors >= plan['policy']['consecutive_source_error_limit']:
                                    halted = 'consecutive_source_errors'
                            else:
                                consecutive_errors = 0
                        except APIError as exc:
                            halted = str(exc)
                        except Exception:
                            halted = 'worker_exception'
                    with budget.lock:
                        fatal_stop = budget.stopped
                    if fatal_stop:
                        halted = halted or 'fatal_backend_error'
                    if Path(plan['stop_file']).exists():
                        halted = halted or 'user_stop_requested'
                    status('stopping' if halted else 'running')
                    if not halted:
                        for _ in done:
                            submit()
                    if halted:
                        with budget.lock:
                            budget.stopped = True
        result = status('complete' if len(receipts) == len(ids) and not halted else 'stopped')
        return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    try:
        result = run_night(read(a.plan), resume=a.resume)
        print(json.dumps(result), flush=True)
        raise SystemExit(0 if result['state'] == 'complete' else 2)
    except Exception as exc:
        print(json.dumps({'state': 'startup_failed', 'error_type': type(exc).__name__}), flush=True)
        raise SystemExit(3)
