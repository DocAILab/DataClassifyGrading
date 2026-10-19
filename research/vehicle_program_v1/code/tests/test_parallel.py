from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from api_backend import APIBudgetExceeded, APIError, default_config
from api_parallel import SharedBudget, run_parallel, scheduler_config
from artifacts import put, read, sha
from backends import ScriptedBackend
from replay import replay
from runner import run
from test_api import MESSAGES, SECRET, response
from test_vehicle import fixture


class ParallelTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = default_config(self.root / 'tokenizer')
        self.config['limits'].update(min_interval_seconds=.000001, max_estimated_cny=4., max_http_requests=100)
        self.scheduler = scheduler_config()
        self.body = {'messages': MESSAGES, 'max_tokens': 512}

    def budget(self, counter=lambda m: 100, **kw):
        return SharedBudget(self.root / 'budget.json', self.config, self.scheduler, counter, **kw)

    def test_sliding_token_window_uses_request_size_and_survives_restart(self):
        self.scheduler['tokens_per_minute'] = 2000
        now = [100.]
        clock = lambda: now[0]
        def sleep(seconds): now[0] += seconds
        budget = self.budget(clock=clock, sleep=sleep)
        for _ in range(2):
            entry = budget.reserve(self.body)
            budget.finish(entry, 'response_received', {'prompt_tokens': 100, 'completion_tokens': 1})
        self.assertEqual(now[0], 100.)
        resumed = SharedBudget(budget.path, self.config, self.scheduler, lambda m: 100, clock=clock, sleep=sleep)
        third = resumed.reserve(self.body)
        self.assertGreaterEqual(third['sent_at_unix'] - 100., 60.1 - 1e-8)
        self.assertEqual(len(read(budget.path)['attempts']), 3)

    def test_success_releases_money_reservation_but_unknown_retains_it(self):
        budget = self.budget()
        entry = budget.reserve(self.body)
        self.assertAlmostEqual(budget.accounted_cny(), .054528)
        budget.finish(entry, 'response_received', {'prompt_tokens': 100, 'completion_tokens': 10})
        self.assertAlmostEqual(budget.accounted_cny(), .00027)
        budget.reserve(self.body)
        self.assertAlmostEqual(budget.accounted_cny(), .054798)
        restarted = self.budget()
        with self.assertRaisesRegex(APIBudgetExceeded, 'shared_backend_stopped'):
            restarted.reserve(self.body)

    def test_concurrent_money_reservations_cannot_overspend(self):
        self.config['limits']['max_estimated_cny'] = .11
        budget = self.budget()
        def reserve():
            try: return budget.reserve(self.body)['number']
            except APIBudgetExceeded: return None
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda _: reserve(), range(20)))
        self.assertEqual(sorted(n for n in results if n is not None), [1, 2])
        self.assertLessEqual(budget.accounted_cny(), .11)

    def test_error_stops_all_workers_without_retry_and_keeps_reserve(self):
        budget = self.budget()
        entry = budget.reserve(self.body)
        budget.finish(entry, 'http_429')
        with self.assertRaises(APIBudgetExceeded): budget.reserve(self.body)
        self.assertAlmostEqual(budget.accounted_cny(), .054528)

    def test_provider_token_estimate_excess_halts(self):
        budget = self.budget()
        entry = budget.reserve(self.body)
        with self.assertRaisesRegex(APIError, 'provider_token_estimate_exceeded'):
            budget.finish(entry, 'response_received', {'prompt_tokens': 1000, 'completion_tokens': 1})
        self.assertTrue(budget.stopped)

    def test_parallel_requests_equal_serial_and_resume_replays_without_gold(self):
        package, output, serial = self.root / 'package', self.root / 'run', self.root / 'serial'
        fixture(package, count=6)
        profile = self.root / 'profile.json'; put(profile, self.config)
        scripted = ScriptedBackend()
        lock = threading.Lock()
        active, peak = [0], [0]
        def transport(url, key, body, timeout):
            with lock:
                active[0] += 1; peak[0] = max(peak[0], active[0])
            try:
                time.sleep(.01)
                return response(scripted(body['messages']))
            finally:
                with lock: active[0] -= 1
        original_open = Path.open
        def guard(path, *args, **kwargs):
            if path.name == 'gold.json': raise AssertionError('inference read gold')
            return original_open(path, *args, **kwargs)
        with patch.object(Path, 'open', guard):
            result = run_parallel(package, output, profile, self.scheduler, counter=lambda m: 100,
                                  key=SECRET, transport=transport)
            run(package, serial, backend='mock')
        self.assertIsNone(result['halted'])
        self.assertEqual(result['committed_count'], 6)
        self.assertEqual(result['model_calls'], 24)
        self.assertEqual(peak[0], 3)
        self.assertEqual(replay(package, output)['completed_sources_replayed'], 6)
        before = {p.as_posix(): sha(p) for p in output.glob('sources/*/receipt.json')}
        def forbidden(*args): raise AssertionError('completed run called API')
        run_parallel(package, output, profile, self.scheduler, resume=True, counter=lambda m: 100,
                     key=SECRET, transport=forbidden)
        self.assertEqual(before, {p.as_posix(): sha(p) for p in output.glob('sources/*/receipt.json')})
        for folder in output.glob('sources/*'):
            actual = [read(p)['messages'] for p in sorted((folder / 'calls').glob('*.json'))]
            expected = [read(p)['messages'] for p in sorted((serial / 'sources' / folder.name / 'calls').glob('*.json'))]
            self.assertEqual(actual, expected)
        for path in output.rglob('*.json'):
            self.assertNotIn(SECRET, path.read_text(encoding='utf-8'))
        self.assertNotIn('SECRET_GOLD', json.dumps([read(p) for p in output.glob('sources/*/calls/*.json')] ))
        changed = deepcopy(self.scheduler); changed['workers'] = 2
        with self.assertRaisesRegex(ValueError, 'run identity differs'):
            run_parallel(package, output, profile, changed, resume=True, key=SECRET, counter=lambda m: 100)

    def test_explicit_selection_and_wall_budget_leave_unstarted_rows_pending(self):
        package, output = self.root / 'package', self.root / 'run'
        fixture(package, count=3)
        profile = self.root / 'profile.json'; put(profile, self.config)
        result = run_parallel(package, output, profile, self.scheduler, positions=[2], counter=lambda m: 100,
                              key=SECRET, transport=lambda url,key,body,timeout: response(ScriptedBackend()(body['messages'])))
        self.assertEqual(result['committed_count'], 1)
        self.assertTrue((output / 'sources/0002/receipt.json').exists())
        self.assertFalse((output / 'sources/0001').exists())
        self.assertEqual(result['pending_count'], 2)
        now = [100.]
        budget = self.budget(clock=lambda: now[0])
        now[0] += 1801
        with self.assertRaisesRegex(APIBudgetExceeded, 'wall_budget_exhausted'): budget.reserve(self.body)

    def test_progress_reads_ledger_under_its_writer_lock_on_windows(self):
        package, output = self.root / 'package', self.root / 'run'
        fixture(package, count=9)
        profile = self.root / 'profile.json'; put(profile, self.config)
        captured = []
        def budget_factory(*args, **kwargs):
            budget = SharedBudget(*args, **kwargs); captured.append(budget); return budget
        def guarded_read(path):
            if Path(path).name == 'api-budget.json':
                self.assertTrue(captured[0].lock._is_owned(), 'ledger opened without its writer lock')
            return read(path)
        with patch('api_parallel.SharedBudget', budget_factory), patch('runner.read', guarded_read):
            result = run_parallel(package, output, profile, self.scheduler, counter=lambda m: 100, key=SECRET,
                                  transport=lambda url,key,body,timeout: response(ScriptedBackend()(body['messages'])))
        self.assertEqual(result['committed_count'], 9)
        self.assertIsNone(result['halted'])


if __name__ == '__main__':
    unittest.main()
