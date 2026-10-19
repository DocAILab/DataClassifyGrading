"""Concurrent independent sources with one durable token, request and money ledger."""
import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from copy import deepcopy
import json
import math
from pathlib import Path
import threading
import time

from api_backend import (APIBackend, APIBudgetExceeded, APIError, Budget, ReferenceCounter,
                         api_generation, api_identity, load_config, read_credential)
from artifacts import (BUDGET, fingerprint, load_controller, load_study, put, read,
                       require_private_output)
from runner import Recorder, environment, exclusive_run, members, read_receipts, write_summary


def scheduler_config():
    return {'workers': 3, 'tokens_per_minute': 72000, 'requests_per_minute': 900,
            'token_margin': 128, 'window_seconds': 60., 'window_guard_seconds': .1}


def validate_scheduler(config):
    if set(config) != set(scheduler_config()):
        raise ValueError('scheduler fields differ')
    for name, value in config.items():
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError('scheduler limits must be positive and finite')
        if name in ('workers', 'tokens_per_minute', 'requests_per_minute', 'token_margin') and type(value) is not int:
            raise ValueError('scheduler counts must be integers')
    if config['workers'] > 8 or config['window_seconds'] != 60.:
        raise ValueError('unsupported scheduler concurrency/window')


class SharedBudget(Budget):
    """Serialize reservations, settle successful usage, retain uncertain maximum charges.

    TPM uses reference input plus margin and maximum output for the entire window;
    it is not refunded after short responses. Unknown outcomes retain full cost.
    """
    def __init__(self, path, config, scheduler, counter, *, clock=time.time, sleep=time.sleep,
                 started_at_unix=None):
        validate_scheduler(scheduler)
        self.path, self.config, self.scheduler = Path(path), config, scheduler
        self.counter, self.clock, self.sleep = counter, clock, sleep
        self.lock = threading.RLock()
        self.stopped = False
        ident = fingerprint({'api': api_identity(config), 'scheduler': scheduler})
        if self.path.exists():
            self.state = read(self.path)
            if self.state['config_sha256'] != ident:
                raise ValueError('shared API budget profile differs')
        else:
            self.state = {'schema': 2, 'config_sha256': ident,
                          'started_at_unix': clock() if started_at_unix is None else started_at_unix, 'attempts': []}
            put(self.path, self.state)
        if started_at_unix is not None and self.state['started_at_unix'] != started_at_unix:
            raise ValueError('shared API budget wall origin differs')
        if [a['number'] for a in self.state['attempts']] != list(range(1, len(self.state['attempts']) + 1)):
            raise ValueError('API budget attempt sequence differs')
        # An uncommitted HTTP outcome must never be retried by restarting the run.
        self.stopped = any(a['state'] != 'response_received' for a in self.state['attempts'])

    def accounted_cny(self):
        return sum(a.get('usage_cost_cny', a['reserved_cny']) for a in self.state['attempts'])

    def check(self, count=1):
        with self.lock:
            if self.stopped:
                raise APIBudgetExceeded('shared_backend_stopped')
            limits = self.config['limits']
            if self.clock() - self.state['started_at_unix'] >= limits['max_wall_seconds']:
                raise APIBudgetExceeded('wall_budget_exhausted')
            if len(self.state['attempts']) + count > limits['max_http_requests']:
                raise APIBudgetExceeded('request_budget_exhausted')
            if self.accounted_cny() + count * self.per_request_upper_cny > limits['max_estimated_cny'] + 1e-12:
                raise APIBudgetExceeded('estimated_cost_budget_exhausted')

    def reservation_delay(self, now):
        """Additional admission delay, evaluated under the reservation lock each iteration."""
        return 0.

    def reserve(self, body):
        with self.lock:
            tokens = self.counter(body['messages']) + body['max_tokens'] + self.scheduler['token_margin']
        if tokens > self.scheduler['tokens_per_minute']:
            raise APIBudgetExceeded('request_exceeds_tpm_window')
        while True:
            with self.lock:
                self.check()
                now = self.clock()
                extra_delay = self.reservation_delay(now)
                window = self.scheduler['window_seconds'] + self.scheduler['window_guard_seconds']
                recent = [a for a in self.state['attempts'] if now - a['sent_at_unix'] < window]
                rate_blocked = (len(recent) >= self.scheduler['requests_per_minute'] or
                                sum(a['rate_tokens'] for a in recent) + tokens > self.scheduler['tokens_per_minute'])
                if extra_delay <= 0 and not rate_blocked:
                    entry = {'number': len(self.state['attempts']) + 1,
                             'reserved_cny': self.per_request_upper_cny,
                             'request_sha256': fingerprint(body), 'state': 'outcome_unknown',
                             'sent_at_unix': now, 'rate_tokens': tokens}
                    self.state['attempts'].append(entry)
                    put(self.path, self.state, replace=True)
                    return entry
                rate_delay = min((a['sent_at_unix'] + window - now for a in recent), default=0.) if rate_blocked else 0.
                delay = max(extra_delay, rate_delay)
            self.sleep(min(max(delay, .001), .5))

    def finish(self, entry, outcome, usage=None):
        with self.lock:
            entry.update(state=outcome, usage=usage, finished_at_unix=self.clock())
            if outcome == 'response_received':
                price = self.config['pricing_cny_per_million']
                cost = (usage['prompt_tokens'] * price['input'] + usage['completion_tokens'] * price['output']) / 1e6
                if cost > entry['reserved_cny'] + 1e-12:
                    self.stopped = True
                    raise APIError('usage_exceeds_cost_reservation')
                entry['usage_cost_cny'] = cost
                # A template difference beyond the margin invalidates the local rate estimate.
                if usage['prompt_tokens'] + 512 > entry['rate_tokens']:
                    entry['rate_tokens'] = usage['prompt_tokens'] + 512
                    self.stopped = True
                    entry['state'] = 'provider_token_estimate_exceeded'
                    put(self.path, self.state, replace=True)
                    raise APIError('provider_token_estimate_exceeded')
            else:
                self.stopped = True
            put(self.path, self.state, replace=True)


def source_receipt(builder, study, row, folder, generator, *, kind='hosted_qwen35_9b_api'):
    folder.mkdir(parents=True)
    put(folder / 'started.json', {'id': row['id'], 'arm': 'NONT-D'})
    recorded = Recorder(generator, folder / 'calls')
    started = time.monotonic()
    receipt = {'id': row['id'], 'status': 'runner_error', 'terminated': False, 'stop_reason': 'runner_error'}
    try:
        controller = builder(recorded, study['registry'], study['rows'], BUDGET, skill_pool='F66')
        if len(controller.library) != 66:
            raise ValueError('F66 library size differs')
        trace = controller.run({'source_id': row['id'], 'domain': row['domain'],
                                'field_name': row['metadata']['field_name']}).to_dict()
        trace['execution_mode'] = kind
        put(folder / 'trace.json', trace)
        receipt.update(status='backend_error' if recorded.failed else 'completed',
                       terminated=trace['terminated'] and not recorded.failed, stop_reason=trace['stop_reason'])
    except Exception as exc:
        receipt['error_type'] = type(exc).__name__
    receipt.update(seconds=time.monotonic() - started, model_calls=len(recorded.calls), files=members(folder),
                   input_tokens=sum(c['generation'].get('input_tokens') or 0 for c in recorded.calls),
                   output_tokens=sum(c['generation'].get('output_tokens') or 0 for c in recorded.calls),
                   token_counts_measured=bool(recorded.calls) and all(
                       c['generation'].get('input_tokens') is not None and c['generation'].get('output_tokens') is not None
                       for c in recorded.calls))
    put(folder / 'receipt.json', receipt)
    return receipt


def run_parallel(package, output, api_config, scheduler, *, positions=None, resume=False,
                 counter=None, key=None, transport=None):
    validate_scheduler(scheduler)
    study, manifest = load_study(package)
    ids = study['expected_ids']
    selected = positions if positions is not None else list(range(1, len(ids) + 1))
    if (not selected or any(type(n) is not int or n < 1 or n > len(ids) for n in selected)
            or len(set(selected)) != len(selected) or selected != sorted(selected)):
        raise ValueError('selected positions must be ordered, unique and within study')
    config = load_config(api_config)
    if config['limits']['min_interval_seconds'] > .001:
        raise ValueError('parallel scheduler replaces fixed request intervals')
    output = Path(output)
    require_private_output(output)
    kind = 'hosted_qwen35_9b_api'
    identity = {'schema': 2, 'arm': 'NONT-D', 'backend': kind,
                'study_sha256': manifest['study_sha256'], 'code_sha256': fingerprint(manifest['code']),
                'model_manifest_sha256': None, 'generation': api_generation(), 'budget': BUDGET,
                'expected_ids': ids, 'environment': environment(), 'api_service': api_identity(config),
                'scheduler': scheduler, 'selected_positions': selected}
    if output.exists() and not resume:
        raise FileExistsError('run exists; explicitly resume same identity')
    output.mkdir(parents=True, exist_ok=resume)
    with exclusive_run(output):
        if (output / 'run.json').exists():
            if read(output / 'run.json') != identity:
                raise ValueError('run identity differs')
        else:
            if set(p.name for p in output.iterdir()) != {'.run.lock'}:
                raise ValueError('unidentified run artifacts')
            put(output / 'run.json', identity)
        receipts = read_receipts(output, ids)
        todo = []
        for n in selected:
            row, folder = study['rows'][n - 1], output / 'sources' / f'{n:04d}'
            if row['id'] in receipts:
                continue
            if folder.exists():
                receipt = {'id': row['id'], 'status': 'interrupted', 'terminated': False,
                           'stop_reason': 'interrupted_before_commit', 'seconds': None,
                           'model_calls': len(list((folder / 'calls').glob('*.json'))), 'files': members(folder)}
                put(folder / 'receipt.json', receipt)
                receipts[row['id']] = receipt
            else:
                todo.append((row, folder))
        halted, budget = None, None
        if todo and any(r['status'] != 'completed' for r in receipts.values()):
            halted = 'existing_source_error'
        elif todo:
            if not (output / 'backend/api-budget.json').exists() and any((output / 'sources').glob('*/calls/*.json')):
                raise ValueError('API budget missing for existing requests')
            counter = counter or ReferenceCounter(config['tokenizer_path'])
            key = key if key is not None else read_credential()
            budget = SharedBudget(output / 'backend/api-budget.json', config, scheduler, counter)
            builder = load_controller()
            def work(row, folder):
                kwargs = {'transport': transport} if transport is not None else {}
                generator = APIBackend(config, output / 'backend', key=key, counter=counter, budget=budget, **kwargs)
                generator.before_source()
                return source_receipt(builder, study, row, folder, generator)
            pending = iter(todo)
            with ThreadPoolExecutor(max_workers=scheduler['workers']) as pool:
                active = {}
                def submit():
                    item = next(pending, None)
                    if item is not None:
                        active[pool.submit(work, *item)] = item[0]['id']
                for _ in range(min(scheduler['workers'], len(todo))):
                    submit()
                while active:
                    done, _ = wait(active, return_when=FIRST_COMPLETED)
                    for future in done:
                        ident = active.pop(future)
                        try:
                            receipt = future.result()
                            receipts[ident] = receipt
                            if receipt['status'] != 'completed':
                                halted = receipt['status']
                        except Exception as exc:
                            halted = type(exc).__name__
                    # Windows may reject replacing an open ledger. Its readers must
                    # share the writer lock, including progress summaries.
                    with budget.lock:
                        write_summary(output, ids, receipts, kind, halted=halted)
                    if not halted:
                        for _ in done:
                            submit()
                    else:
                        with budget.lock:
                            budget.stopped = True
        if budget is not None:
            with budget.lock:
                result = write_summary(output, ids, receipts, kind, halted=halted)
        else:
            result = write_summary(output, ids, receipts, kind, halted=halted)
        path = output / 'backend/api-budget.json'
        attempts = read(path)['attempts'] if path.exists() else []
        result.update(selected_count=len(selected), selected_committed_count=sum(ids[n - 1] in receipts for n in selected),
                      usage_estimate_plus_uncertain_reserves_cny=sum(a.get('usage_cost_cny', a['reserved_cny']) for a in attempts),
                      selected_complete=all(ids[n - 1] in receipts for n in selected))
        put(output / 'summary.json', result, replace=True)
        return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--api-config', type=Path, required=True)
    p.add_argument('--scheduler', type=Path, required=True)
    p.add_argument('--start-position', type=int, default=1)
    p.add_argument('--count', type=int)
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    positions = list(range(a.start_position, a.start_position + a.count)) if a.count is not None else None
    result = run_parallel(a.package, a.output, a.api_config, read(a.scheduler), positions=positions, resume=a.resume)
    print(json.dumps(result))
    raise SystemExit(0 if result['selected_complete'] and not result['halted'] else 2)
