"""Adversarial recovery and admission tests: no real HTTP or credentials."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from artifacts import put, read, sha
from api_backend import APIBudgetExceeded, default_config
from api_parallel import scheduler_config
from backends import ScriptedBackend
from night_ops import preflight
from night_runner import NightBudget, run_night
from night_supervisor import supervise
from night_resume import effective_cooldown
from test_api import MESSAGES, response
from test_night import night_fixture


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.network = patch('urllib.request.OpenerDirector.open', side_effect=AssertionError('real network forbidden'))
        self.network.start(); self.addCleanup(self.network.stop)

    def paused_run(self, root):
        root.mkdir(exist_ok=True)
        plan = night_fixture(root, count=3)
        plan['scheduler']['workers'] = 1
        calls = [0]
        def transport(url, key, body, timeout):
            calls[0] += 1
            if calls[0] == 4:
                Path(plan['stop_file']).touch()
            return response(ScriptedBackend()(body['messages']))
        first = run_night(plan, transport=transport)
        self.assertEqual(first['committed_sources'], 1)
        Path(plan['stop_file']).unlink()
        return plan, Path(plan['output']) / 'backend/api-budget.json'

    def assert_rejected(self, plan):
        with patch('night_runner.read_credential', side_effect=AssertionError('credential read')), \
                patch('night_runner.APIBackend', side_effect=AssertionError('backend initialized')):
            with self.assertRaises(RuntimeError): preflight(plan, resume=True)
            with self.assertRaises(RuntimeError): run_night(plan, resume=True)

    def test_missing_ledger_rejected_without_recreating_or_requesting(self):
        plan, ledger = self.paused_run(self.root)
        before = sha(Path(plan['output']) / 'run.json')
        ledger.rename(self.root / 'preserved-ledger.json')
        self.assert_rejected(plan)
        self.assertFalse(ledger.exists())
        self.assertEqual(before, sha(Path(plan['output']) / 'run.json'))

    def test_resume_without_existing_run_never_initializes(self):
        plan = night_fixture(self.root)
        with self.assertRaises(RuntimeError): preflight(plan, resume=True)
        with self.assertRaisesRegex(RuntimeError, 'resume_run_identity_missing'): run_night(plan, resume=True)
        self.assertFalse(Path(plan['output']).exists())

    def test_rollback_hash_usage_charge_and_origin_corruption_rejected(self):
        def mutate(value, kind):
            if kind == 'empty': value['attempts'] = []
            elif kind == 'truncated': value['attempts'].pop()
            elif kind == 'hash': value['attempts'][0]['request_sha256'] = '0' * 64
            elif kind == 'usage': value['attempts'][0]['usage']['prompt_tokens'] += 1
            elif kind == 'cost': value['attempts'][0]['usage_cost_cny'] = 0.
            elif kind == 'origin': value['started_at_unix'] += 1
            elif kind == 'unmatched':
                extra = deepcopy(value['attempts'][0]); extra['number'] = 5; value['attempts'].append(extra)
        for kind in ('empty', 'truncated', 'hash', 'usage', 'cost', 'origin', 'unmatched'):
            with self.subTest(kind=kind):
                plan, ledger = self.paused_run(self.root / kind)
                state = read(ledger); mutate(state, kind); put(ledger, state, replace=True)
                self.assert_rejected(plan)

    def test_normal_resume_preserves_cost_origin_and_receipt(self):
        plan, ledger = self.paused_run(self.root)
        original = read(ledger)
        receipt = Path(plan['output']) / 'sources/0001/receipt.json'
        digest = sha(receipt)
        self.assertTrue(preflight(plan, resume=True)['ready'])
        result = run_night(plan, resume=True)
        after = read(ledger)
        self.assertEqual(result['state'], 'complete')
        self.assertEqual(after['started_at_unix'], original['started_at_unix'])
        self.assertEqual(after['attempts'][:4], original['attempts'])
        self.assertEqual(len(after['attempts']), 12)
        self.assertAlmostEqual(result['usage_plus_uncertain_reserves_cny'], .00324)
        self.assertEqual(digest, sha(receipt))

    def test_expired_resume_rejected_before_supervisor_network_or_worker(self):
        plan, ledger = self.paused_run(self.root)
        plan_path = self.root / 'plan.json'; put(plan_path, plan)
        origin = read(ledger)['started_at_unix']
        with patch('time.time', return_value=origin + 601), \
                patch('night_supervisor.subprocess.Popen', side_effect=AssertionError('worker started')):
            with self.assertRaisesRegex(RuntimeError, 'wall_budget_exhausted'): preflight(plan, resume=True)
            with self.assertRaisesRegex(RuntimeError, 'wall_budget_exhausted'): supervise(plan_path, resume=True)

    def test_cooldown_added_during_tpm_wait_is_honored(self):
        for outcome, delay in [('http_429', 120.), ('http_503', 30.)]:
            with self.subTest(outcome=outcome):
                config = default_config(self.root)
                config['limits'].update(max_estimated_cny=10., max_wall_seconds=600.)
                scheduler = scheduler_config(); scheduler['tokens_per_minute'] = 1000
                now = [1000.]; holder = {}; triggered = [False]
                def sleep(seconds):
                    # Inject a new cooldown just before the token window opens.
                    if not triggered[0] and now[0] >= 1059.:
                        triggered[0] = True
                        holder['budget'].finish(holder['first'], outcome)
                    now[0] += seconds
                budget = NightBudget(self.root / (outcome + '.json'), config, scheduler, lambda m: 100,
                    policy={'rate_limit_cooldown_seconds': 120., 'transient_cooldown_seconds': 30.},
                    stop_file=self.root / 'STOP', clock=lambda: now[0], sleep=sleep)
                holder['budget'] = budget
                body = {'messages': MESSAGES, 'max_tokens': 512}
                holder['first'] = budget.reserve(body)
                second = budget.reserve(body)
                self.assertTrue(triggered[0])
                self.assertGreaterEqual(second['sent_at_unix'], 1059. + delay)
                self.assertGreaterEqual(second['sent_at_unix'], budget.cooldown_until)

    def test_resume_preserves_unexpired_cooldown_and_stop_interrupts_wait(self):
        config = default_config(self.root); config['limits'].update(max_estimated_cny=10.)
        scheduler = scheduler_config(); now = [1000.]
        policy = {'rate_limit_cooldown_seconds': 120., 'transient_cooldown_seconds': 30.}
        body = {'messages': MESSAGES, 'max_tokens': 512}
        def sleep(seconds): now[0] += seconds
        kwargs = dict(policy=policy, stop_file=self.root / 'STOP', clock=lambda: now[0], sleep=sleep)
        budget = NightBudget(self.root / 'ledger.json', config, scheduler, lambda m: 100, **kwargs)
        entry = budget.reserve(body); budget.finish(entry, 'http_429')
        restored = NightBudget(budget.path, config, scheduler, lambda m: 100, **kwargs)
        self.assertEqual(restored.cooldown_until, 1120.)
        self.assertGreaterEqual(restored.reserve(body)['sent_at_unix'], 1120.)
        restored.cooldown_until = now[0] + 120.
        def stop(seconds): Path(kwargs['stop_file']).touch(); now[0] += seconds
        restored.sleep = stop
        with self.assertRaisesRegex(APIBudgetExceeded, 'user_stop_requested'): restored.reserve(body)
        self.assertEqual(len(restored.state['attempts']), 2)

    def test_failure_drained_after_circuit_stop_restores_conservative_cooldown(self):
        config = default_config(self.root); config['limits'].update(max_estimated_cny=10.)
        scheduler = scheduler_config(); now = [1000.]
        policy = {'rate_limit_cooldown_seconds': 120., 'transient_cooldown_seconds': 30.}
        body = {'messages': MESSAGES, 'max_tokens': 512}
        def sleep(seconds): now[0] += seconds
        kwargs = dict(policy=policy, stop_file=self.root / 'STOP', clock=lambda: now[0], sleep=sleep)
        budget = NightBudget(self.root / 'ledger.json', config, scheduler, lambda m: 100, **kwargs)
        first, tail = budget.reserve(body), budget.reserve(body)
        budget.finish(first, 'http_503')
        budget.stopped = True
        now[0] = 1008.478
        budget.finish(tail, 'transport_or_json_error')
        original = read(budget.path)
        self.assertEqual(original['cooldown_until_unix'], 1030.)
        restored = NightBudget(budget.path, config, scheduler, lambda m: 100, **kwargs)
        self.assertEqual(read(budget.path), original)
        self.assertAlmostEqual(restored.cooldown_until, 1038.478)
        next_entry = restored.reserve(body)
        self.assertGreaterEqual(next_entry['sent_at_unix'], 1038.478)

    def test_effective_cooldown_uses_longer_recorded_or_429_tail_deadline(self):
        policy = {'rate_limit_cooldown_seconds': 120., 'transient_cooldown_seconds': 30.}
        ledger = {'cooldown_until_unix': 1050., 'attempts': [{'state': 'http_429', 'finished_at_unix': 1000.}]}
        self.assertEqual(effective_cooldown(ledger, policy), 1120.)
        ledger['cooldown_until_unix'] = 1200.
        self.assertEqual(effective_cooldown(ledger, policy), 1200.)
        for invalid in (float('nan'), -1, '1000'):
            with self.subTest(invalid=invalid):
                ledger['attempts'][0]['finished_at_unix'] = invalid
                with self.assertRaises(RuntimeError): effective_cooldown(ledger, policy)


if __name__ == '__main__': unittest.main()
