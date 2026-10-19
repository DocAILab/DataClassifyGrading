from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from api_backend import APIBudgetExceeded, APIError, default_config
from api_parallel import scheduler_config
from artifacts import put, read, sha
from backends import ScriptedBackend
from night_ops import preflight
from night_runner import NightBudget, run_night
from night_supervisor import keep_awake
from replay import replay
from runner import run
from test_api import MESSAGES, response
from test_vehicle import fixture


def night_fixture(root, *, count=6, seed_count=0):
    package, home = root / 'package', root / 'home'
    study = fixture(package, count=count)
    config = default_config(root / 'fictional-tokenizer')
    config['limits'].update(min_interval_seconds=.000001, max_http_requests=count * 16,
                            max_estimated_cny=10., max_wall_seconds=600.)
    put(package / 'night-api-profile.json', config)
    provenance = {'receipts': {}, 'usage_plus_uncertain_reserves_cny': .1 if seed_count else 0.}
    if seed_count:
        donor = root / 'donor'
        run(package, donor, max_samples=seed_count)
        import shutil
        for n, ident in enumerate(study['expected_ids'][:seed_count], 1):
            src = donor / f'sources/{n:04d}'
            if n == 1:
                receipt = read(src / 'receipt.json'); receipt.update(status='backend_error', terminated=False)
                put(src / 'receipt.json', receipt, replace=True)
            shutil.copytree(src, package / f'seed/sources/{n:04d}')
            provenance['receipts'][ident] = {'receipt_sha256': sha(src / 'receipt.json')}
    put(package / 'seed-provenance.json', provenance)
    home.mkdir()
    scheduler = scheduler_config(); scheduler.update(workers=2, tokens_per_minute=100000000, requests_per_minute=100000)
    plan = {'schema': 'vehicle-nontd-night-v1', 'mode': 'scripted_test', 'package': str(package),
            'package_manifest_sha256': sha(package / 'manifest.json'),
            'home': str(home), 'output': str(home / 'run'), 'stop_file': str(home / 'STOP'),
            'files': {n: sha(package / n) for n in ('night-api-profile.json', 'seed-provenance.json')},
            'scheduler': scheduler, 'supervisor_wall_seconds': 30.,
            'policy': {'heartbeat_seconds': .01, 'transient_cooldown_seconds': .001,
                       'rate_limit_cooldown_seconds': .001, 'consecutive_source_error_limit': 3,
                       'shutdown_grace_seconds': 1.}}
    return plan


class NightTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_preserve_seed_errors_continue_fresh_sources_and_replay(self):
        plan = night_fixture(self.root, count=8, seed_count=2)
        before = {p.name + p.parent.name: sha(p) for p in (self.root / 'package/seed').glob('sources/*/receipt.json')}
        original_open = Path.open
        def guard(path, *a, **kw):
            if path.name in ('gold.json', '敏感信息交接.md'): raise AssertionError('inference read gold or credential')
            return original_open(path, *a, **kw)
        with patch.object(Path, 'open', guard):
            result = run_night(plan)
        self.assertEqual(result['state'], 'complete')
        self.assertEqual(result['newly_finished_sources'], 6)
        self.assertEqual(result['new_http_attempts'], 24)
        self.assertEqual(result['source_errors'], 1)
        self.assertEqual(result['carried_usage_plus_uncertain_reserves_cny'], .1)
        after = {p.name + p.parent.name: sha(p) for p in Path(plan['output']).glob('sources/000[12]/receipt.json')}
        self.assertEqual(before, after)
        self.assertEqual(replay(plan['package'], plan['output'])['completed_sources_replayed'], 7)
        previous = {p.as_posix(): sha(p) for p in Path(plan['output']).glob('sources/*/receipt.json')}
        resumed = run_night(plan, resume=True)
        self.assertEqual(resumed['new_http_attempts'], 24)
        self.assertEqual(previous, {p.as_posix(): sha(p) for p in Path(plan['output']).glob('sources/*/receipt.json')})

    def test_one_transport_failure_is_not_retried_and_new_sources_continue(self):
        plan = night_fixture(self.root, count=5)
        calls = [0]
        def transport(url, key, body, timeout):
            calls[0] += 1
            if calls[0] == 1: raise APIError('transport_or_json_error')
            return response(ScriptedBackend()(body['messages']))
        import time
        def slow_error_commit(path, value, **kwargs):
            if Path(path).name == 'api-budget.json' and any(a['state'] == 'transport_or_json_error' for a in value['attempts']):
                time.sleep(.005)
            return put(path, value, **kwargs)
        with patch('api_parallel.put', slow_error_commit):
            result = run_night(plan, transport=transport)
        self.assertEqual(result['state'], 'complete')
        self.assertEqual(result['source_errors'], 1)
        self.assertEqual(calls[0], 17)
        self.assertEqual(result['uncertain_new_requests'], 1)
        self.assertGreaterEqual(result['usage_plus_uncertain_reserves_cny'], .054528)

    def test_three_consecutive_source_errors_trip_circuit(self):
        plan = night_fixture(self.root, count=9)
        def failure(*args): raise APIError('transport_or_json_error')
        result = run_night(plan, transport=failure)
        self.assertEqual(result['state'], 'stopped')
        self.assertGreaterEqual(result['pending_sources'], 5)
        self.assertIn(result['halted'], ('consecutive_source_errors', 'fatal_backend_error'))
        self.assertLessEqual(result['new_http_attempts'], 4)

    def test_auth_error_stops_all_new_requests(self):
        plan = night_fixture(self.root, count=8)
        def failure(*args): raise APIError('http_401')
        result = run_night(plan, transport=failure)
        self.assertEqual(result['state'], 'stopped')
        self.assertEqual(result['halted'], 'fatal_backend_error')
        self.assertLessEqual(result['new_http_attempts'], 2)

    def test_stop_marker_blocks_http_and_retains_pending_sources(self):
        plan = night_fixture(self.root)
        Path(plan['stop_file']).touch()
        result = run_night(plan)
        self.assertEqual(result['new_http_attempts'], 0)
        self.assertEqual(result['pending_sources'], 6)
        self.assertEqual(result['halted'], 'user_stop_requested')

    def test_money_cap_includes_carried_pilot_spend(self):
        plan = night_fixture(self.root, seed_count=1)
        profile = Path(plan['package']) / 'night-api-profile.json'
        config = read(profile); config['limits']['max_estimated_cny'] = .95
        put(profile, config, replace=True); plan['files']['night-api-profile.json'] = sha(profile)
        result = run_night(plan)
        self.assertEqual(result['new_http_attempts'], 0)
        self.assertEqual(result['halted'], 'estimated_cost_budget_exhausted')

    def test_unknown_durable_request_blocks_restart_and_preflight(self):
        plan = night_fixture(self.root)
        config = read(Path(plan['package']) / 'night-api-profile.json')
        ledger = NightBudget(self.root / 'budget.json', config, plan['scheduler'], lambda m: 100,
                             policy=plan['policy'], stop_file=plan['stop_file'])
        ledger.reserve({'messages': MESSAGES, 'max_tokens': 512})
        restored = NightBudget(self.root / 'budget.json', config, plan['scheduler'], lambda m: 100,
                               policy=plan['policy'], stop_file=plan['stop_file'])
        with self.assertRaises(APIBudgetExceeded): restored.check()
        Path(plan['output']).mkdir()
        put(Path(plan['output']) / 'backend/api-budget.json', read(self.root / 'budget.json'))
        with self.assertRaisesRegex(RuntimeError, 'uncertain_or_fatal_attempt_requires_review'):
            preflight(plan, resume=True)

    def test_sleep_guard_releases_system_requirement_without_display_flag(self):
        import os
        if os.name != 'nt': self.skipTest('Windows sleep API')
        calls = []
        def fake(flags): calls.append(flags); return 1
        with patch('night_supervisor.ctypes.windll.kernel32.SetThreadExecutionState', fake):
            with self.assertRaises(RuntimeError):
                with keep_awake(): raise RuntimeError('simulated worker exception')
        self.assertEqual(calls, [0x80000001, 0x80000000])

    def test_atomic_windows_commit_retry_preserves_evidence(self):
        original = Path.replace; calls = [0]
        def transient(path, target):
            calls[0] += 1
            if calls[0] < 4: raise PermissionError('simulated status reader')
            return original(path, target)
        with patch.object(Path, 'replace', transient), patch('artifacts.time.sleep'):
            put(self.root / 'committed.json', {'synthetic': True})
        self.assertEqual(read(self.root / 'committed.json'), {'synthetic': True})
        self.assertEqual(calls[0], 4)


if __name__ == '__main__': unittest.main()
