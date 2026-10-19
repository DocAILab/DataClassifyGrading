from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import urllib.error
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from api_backend import (APIBackend, APIBudgetExceeded, APIError, Budget, MODEL,
                         NoRedirect, api_identity, credential_from_text, default_config,
                         http_post, load_config)
from artifacts import put, read
from backends import ScriptedBackend
from replay import replay
from runner import run
from test_vehicle import fixture

SECRET = 'fictional-secret-for-unit-tests-only'
MESSAGES = [{'role': 'system', 'content': 'Return JSON.'}, {'role': 'user', 'content': 'synthetic'}]


def response(content='{"ok":true}', *, inp=100, out=10, reason='stop'):
    return {'model': MODEL, 'usage': {'prompt_tokens': inp, 'completion_tokens': out},
            'choices': [{'message': {'content': content}, 'finish_reason': reason}]}


class APITests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = default_config(self.root / 'fictional-tokenizer')
        self.config['limits']['min_interval_seconds'] = .00001
        self.config['limits']['max_estimated_cny'] = 10.

    def backend(self, transport=None, counter=lambda m: 100, **kwargs):
        return APIBackend(self.config, self.root / 'backend', key=SECRET,
                          counter=counter, transport=transport or (lambda *a: response()), **kwargs)

    def test_no_key_in_profile_identity_and_wrong_model_or_host_refused(self):
        self.assertNotIn(SECRET, json.dumps(api_identity(self.config)))
        for key, value in [('model', 'qwen3.5-flash'), ('base_url', 'https://example.com/v1'),
                           ('provider_precision', 'FP16'), ('api_key', SECRET)]:
            config = deepcopy(self.config); config[key] = value
            path = self.root / (key + '.json'); put(path, config)
            with self.assertRaises(ValueError):
                load_config(path)

    def test_contract_sent_to_same_9b_without_json_mode_or_gold(self):
        calls = []
        def transport(url, key, body, timeout):
            calls.append(deepcopy(body))
            self.assertEqual(url, 'https://api.siliconflow.cn/v1/chat/completions')
            self.assertEqual(key, SECRET)
            return response()
        backend = self.backend(transport)
        self.assertEqual(backend(MESSAGES), '{"ok":true}')
        self.assertEqual(calls[0]['model'], MODEL)
        self.assertIs(calls[0]['enable_thinking'], False)
        self.assertNotIn('seed', calls[0])
        self.assertNotIn('chat_template_kwargs', calls[0])
        self.assertEqual(calls[0]['max_tokens'], 512)
        self.assertEqual(calls[0]['temperature'], 0.)
        self.assertNotIn('response_format', calls[0])
        self.assertNotIn(SECRET, (self.root / 'backend/api-budget.json').read_text())

    def test_overlong_reference_context_prevents_any_http(self):
        def forbidden(*args):
            raise AssertionError('network called for overlong prompt')
        backend = self.backend(forbidden, counter=lambda m: 32257)
        with self.assertRaisesRegex(APIError, 'reference_context_budget_exceeded'):
            backend(MESSAGES)
        self.assertEqual(backend.budget.state['attempts'], [])

    def test_durable_unknown_attempt_is_counted_and_never_retried(self):
        calls = []
        def fail(*args):
            calls.append(1)
            raise TimeoutError(SECRET)
        backend = self.backend(fail)
        with self.assertRaisesRegex(APIError, '^api_contract_error$'):
            backend(MESSAGES)
        self.assertEqual(len(calls), 1)
        resumed = self.backend()
        self.assertEqual(len(resumed.budget.state['attempts']), 1)
        self.assertNotIn(SECRET, (self.root / 'backend/api-budget.json').read_text())
        self.assertNotIn(SECRET, json.dumps(backend.last_generation_metadata))

    def test_actual_http_error_body_is_not_propagated(self):
        class FakeOpener:
            def open(self, *args, **kwargs):
                raise urllib.error.HTTPError('https://api.together.ai', 401, SECRET, {}, io.BytesIO(SECRET.encode()))
        with patch('urllib.request.build_opener', return_value=FakeOpener()):
            with self.assertRaisesRegex(APIError, '^http_401$'):
                http_post('https://api.together.ai/v1/chat/completions', SECRET, {}, 1)

    def test_redirect_does_not_forward_bearer(self):
        with self.assertRaisesRegex(APIError, '^redirect_refused$'):
            NoRedirect().redirect_request(None, None, 307, '', {}, 'https://example.com')

    def test_wrong_model_thinking_missing_usage_and_overbudget_usage_stop(self):
        bad = []
        value = response(); value['model'] = 'Qwen/Qwen3.5-27B'; bad.append(value)
        value = response(); value['choices'][0]['message']['reasoning_content'] = 'hidden'; bad.append(value)
        value = response(); value['usage'] = {}; bad.append(value)
        bad.append(response(inp=32257)); bad.append(response(out=513)); bad.append(response(SECRET))
        for i, value in enumerate(bad):
            backend = APIBackend(self.config, self.root / f'bad-{i}', key=SECRET, counter=lambda m: 100,
                                 transport=lambda *a, v=value: v)
            with self.assertRaises(APIError):
                backend(MESSAGES)
            self.assertTrue(backend.last_generation_metadata['error'])

    def test_length_finish_reason_is_preserved_for_controller(self):
        backend = self.backend(lambda *a: response('{"unfinished":', out=512, reason='length'))
        self.assertEqual(backend(MESSAGES), '{"unfinished":')
        self.assertTrue(backend.last_generation_metadata['truncation_at_budget'])

    def test_request_cost_and_wall_limits_include_restart(self):
        for kind in ('requests', 'cost', 'wall'):
            config = deepcopy(self.config)
            if kind == 'requests': config['limits']['max_http_requests'] = 1
            if kind == 'cost': config['limits']['max_estimated_cny'] = .0001
            clock = [10.]
            budget = Budget(self.root / f'{kind}.json', config, clock=lambda: clock[0])
            if kind == 'requests': budget.reserve({'synthetic': 1})
            if kind == 'wall': clock[0] += 1801
            resumed = Budget(self.root / f'{kind}.json', config, clock=lambda: clock[0])
            with self.assertRaises(APIBudgetExceeded): resumed.reserve({'synthetic': 2})

    def test_credential_loader_requires_unique_dedicated_section(self):
        text = '## Other\nAPI Key: other-key-must-not-be-used\n## 汽车 NONT-D SiliconFlow API\nAPI Key: ' + SECRET
        self.assertEqual(credential_from_text(text), SECRET)
        for value in ('## Other\nAPI Key: ' + SECRET, text + '\n' + text,
                      '## 汽车 NONT-D SiliconFlow API\nAPI Key: <尚未填写>'):
            with self.assertRaises(ValueError): credential_from_text(value)

    def test_api_integration_gold_isolation_resume_and_full_replay(self):
        package, output = self.root / 'package', self.root / 'run'
        fixture(package)
        config_path = self.root / 'profile.json'; put(config_path, self.config)
        scripted = ScriptedBackend()
        observed = []
        def transport(url, key, body, timeout):
            observed.append(body)
            return response(scripted(body['messages']))
        def factory():
            return APIBackend(self.config, output / 'backend', key=SECRET,
                              counter=lambda m: 100, transport=transport)
        original_open = Path.open
        def guard(path, *args, **kwargs):
            if path.name == 'gold.json': raise AssertionError('inference read gold')
            return original_open(path, *args, **kwargs)
        with patch.object(Path, 'open', guard):
            first = run(package, output, backend='api', api_config=config_path, factory=factory, max_samples=1)
            self.assertEqual(first['committed_count'], 1)
            complete = run(package, output, backend='api', api_config=config_path, factory=factory, resume=True)
        self.assertEqual(complete['committed_count'], 3)
        self.assertEqual(complete['api_http_attempts'], 12)
        self.assertEqual(replay(package, output)['completed_sources_replayed'], 3)
        for path in output.rglob('*.json'):
            self.assertNotIn(SECRET, path.read_text(encoding='utf-8'))
        self.assertNotIn('SECRET_GOLD', json.dumps(observed))
        def forbidden(): raise AssertionError('completed run initialized API')
        run(package, output, backend='api', api_config=config_path, factory=forbidden, resume=True)
        changed = deepcopy(self.config); changed['limits']['max_estimated_cny'] = 2.
        put(self.root / 'changed.json', changed)
        with self.assertRaisesRegex(ValueError, 'run identity differs'):
            run(package, output, backend='api', api_config=self.root / 'changed.json', resume=True)

    def test_source_is_left_pending_when_its_call_allowance_does_not_fit(self):
        package, output = self.root / 'package', self.root / 'run'
        fixture(package)
        self.config['limits']['max_http_requests'] = 15
        path = self.root / 'profile.json'; put(path, self.config)
        result = run(package, output, backend='api', api_config=path,
                     factory=lambda: APIBackend(self.config, output / 'backend', key=SECRET,
                                                counter=lambda m: 100, transport=lambda *a: response()))
        self.assertEqual(result['halted'], 'APIBudgetExceeded')
        self.assertEqual(result['pending_count'], 3)
        self.assertEqual(result['committed_count'], 0)
        self.assertEqual(result['api_http_attempts'], 0)


if __name__ == '__main__':
    unittest.main()
