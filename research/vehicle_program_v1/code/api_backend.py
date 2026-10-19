"""Bounded text-only hosted Qwen3.5-9B inference; credentials never enter artifacts."""
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import re
import time
import urllib.error
import urllib.request

from artifacts import BUDGET, GENERATION, MODEL_REVISION, fingerprint, put, read, sha

MODEL = 'Qwen/Qwen3.5-9B'
TOKENIZER_FILES = {
    'chat_template.jinja': 'a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715',
    'config.json': 'd0883072e01861ed0b2d47be3c16c36a8e81c224c7ffaa310c6558fb3f932b05',
    'merges.txt': 'a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d',
    'tokenizer.json': '5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42',
    'tokenizer_config.json': '316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8',
    'vocab.json': 'ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003',
}
CREDENTIAL_FILE = Path(os.environ.get('VEHICLE_NONTD_CREDENTIAL_FILE', str(Path.home() / '.config/vehicle-nontd/credentials.md')))
CREDENTIAL_HEADING = '汽车 NONT-D SiliconFlow API'


class APIError(RuntimeError):
    """Only fixed error codes are public; never include server error bodies."""


class APIBudgetExceeded(APIError):
    pass


def default_config(tokenizer):
    return {'schema': 'vehicle-nontd-api-v2', 'provider': 'siliconflow',
            'base_url': 'https://api.siliconflow.cn/v1', 'model': MODEL,
            'provider_precision': 'not_verified', 'provider_revision': 'not_exposed',
            'tokenizer_path': str(Path(tokenizer).resolve()),
            'pricing_cny_per_million': {'input': 1.5, 'output': 12.},
            'limits': {'max_http_requests': 32, 'max_estimated_cny': 1.,
                       'min_interval_seconds': 30., 'request_timeout_seconds': 90.,
                       'max_wall_seconds': 1800.}}


def load_config(path):
    config = read(path)
    expected = set(default_config('.'))
    if set(config) != expected or config['schema'] != 'vehicle-nontd-api-v2':
        raise ValueError('API profile fields differ; credentials are not profile fields')
    if config['provider'] != 'siliconflow' or config['model'] != MODEL:
        raise ValueError('this profile supports only the named hosted Qwen3.5-9B service')
    # Credentials must never be forwarded to an unreviewed host or redirect target.
    if config['base_url'] != 'https://api.siliconflow.cn/v1':
        raise ValueError('API endpoint differs from the reviewed provider')
    if config['provider_precision'] != 'not_verified' or config['provider_revision'] != 'not_exposed':
        raise ValueError('do not claim hosted FP16 or a verified HF weight revision')
    limits = config['limits']
    if set(limits) != set(default_config('.')['limits']):
        raise ValueError('invalid API limit fields')
    for value in [*limits.values(), *config['pricing_cny_per_million'].values()]:
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError('API limits and prices must be finite and positive')
    if type(limits['max_http_requests']) is not int:
        raise ValueError('request limit must be an integer')
    if set(config['pricing_cny_per_million']) != {'input', 'output'}:
        raise ValueError('invalid price fields')
    return config


def api_identity(config):
    public = {k: deepcopy(v) for k, v in config.items() if k != 'tokenizer_path'}
    public.update(reference_tokenizer={'repo_id': MODEL, 'revision': MODEL_REVISION,
                                      'files': TOKENIZER_FILES},
                  hosted_weight_hash_verified=False, local_fp16_equivalence_claim=False)
    return public


def api_generation():
    return {'seed_requested': None, 'seed_support': 'not_documented_by_provider', 'temperature': 0.,
            'context_cap': GENERATION['context_cap'], 'max_new_tokens': GENERATION['max_new_tokens'],
            'enable_thinking': False, 'provider_precision': 'not_verified',
            'deterministic_output_guaranteed': False}


class ReferenceCounter:
    """Offline official template/tokenizer, not proof of the provider's exact template."""
    def __init__(self, folder):
        folder = Path(folder)
        for name, expected in TOKENIZER_FILES.items():
            if (folder / name).is_symlink() or sha(folder / name) != expected:
                raise ValueError('reference tokenizer identity differs')
        from tokenizers import Tokenizer
        from jinja2.sandbox import ImmutableSandboxedEnvironment
        self.tokenizer = Tokenizer.from_file(str(folder / 'tokenizer.json'))
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        def invalid_template(message):
            raise ValueError('reference chat template rejected messages')
        env.globals['raise_exception'] = invalid_template
        self.template = env.from_string((folder / 'chat_template.jinja').read_text(encoding='utf-8'))
        self.settings = read(folder / 'tokenizer_config.json')

    def __call__(self, messages):
        prompt = self.template.render(messages=messages, add_generation_prompt=True,
                                      enable_thinking=False, **{k: self.settings.get(k)
                                                               for k in ('bos_token', 'eos_token', 'pad_token')})
        return len(self.tokenizer.encode(prompt, add_special_tokens=False).ids)


def credential_from_text(text, heading=CREDENTIAL_HEADING):
    lines = text.splitlines()
    selected = []
    active = False
    matches = 0
    for line in lines:
        if re.match(r'^#{1,6}\s', line):
            active = line.lstrip('#').strip() == heading
            matches += int(active)
        elif active:
            selected.append(line)
    keys = [m.group(1).strip() for line in selected
            if (m := re.fullmatch(r'\s*API Key\s*[:：]\s*(\S+)\s*', line))]
    if matches != 1 or len(keys) != 1 or len(keys[0]) < 16 or any(c in keys[0] for c in '<>`'):
        raise ValueError('one API Key line is required in the dedicated credential section')
    return keys[0]


def read_credential():
    return credential_from_text(CREDENTIAL_FILE.read_text(encoding='utf-8'))


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise APIError('redirect_refused')


def http_post(url, key, body, timeout):
    request = urllib.request.Request(url, data=json.dumps(body, ensure_ascii=False).encode('utf-8'),
                                     headers={'Authorization': 'Bearer ' + key,
                                              'Content-Type': 'application/json'}, method='POST')
    try:
        with urllib.request.build_opener(NoRedirect()).open(request, timeout=timeout) as response:
            data = response.read(8 * 1024 * 1024 + 1)
            if len(data) > 8 * 1024 * 1024:
                raise APIError('response_too_large')
            return json.loads(data)
    except urllib.error.HTTPError as exc:
        code = exc.code
        exc.close()
        raise APIError('http_' + str(code)) from None
    except APIError:
        raise
    except Exception:
        raise APIError('transport_or_json_error') from None


class Budget:
    """Reserve worst-case cost durably before every HTTP attempt, with no automatic retries."""
    def __init__(self, path, config, *, clock=time.time):
        self.path, self.config, self.clock = Path(path), config, clock
        if self.path.exists():
            self.state = read(self.path)
            if self.state['config_sha256'] != fingerprint(api_identity(config)):
                raise ValueError('API budget belongs to a different profile')
        else:
            self.state = {'schema': 1, 'config_sha256': fingerprint(api_identity(config)),
                          'started_at_unix': clock(), 'attempts': []}
            put(self.path, self.state)
        if [a['number'] for a in self.state['attempts']] != list(range(1, len(self.state['attempts']) + 1)):
            raise ValueError('API budget attempt sequence differs')

    @property
    def per_request_upper_cny(self):
        p = self.config['pricing_cny_per_million']
        return ((GENERATION['context_cap'] - GENERATION['max_new_tokens']) * p['input']
                + GENERATION['max_new_tokens'] * p['output']) / 1_000_000

    def check(self, count=1):
        limits = self.config['limits']
        if self.clock() - self.state['started_at_unix'] >= limits['max_wall_seconds']:
            raise APIBudgetExceeded('wall_budget_exhausted')
        if len(self.state['attempts']) + count > limits['max_http_requests']:
            raise APIBudgetExceeded('request_budget_exhausted')
        cost = sum(a['reserved_cny'] for a in self.state['attempts'])
        if cost + count * self.per_request_upper_cny > limits['max_estimated_cny'] + 1e-12:
            raise APIBudgetExceeded('estimated_cost_budget_exhausted')

    def reserve(self, body):
        self.check()
        entry = {'number': len(self.state['attempts']) + 1, 'reserved_cny': self.per_request_upper_cny,
                 'request_sha256': fingerprint(body), 'state': 'outcome_unknown',
                 'sent_at_unix': self.clock()}
        self.state['attempts'].append(entry)
        put(self.path, self.state, replace=True)
        return entry

    def finish(self, entry, outcome, usage=None):
        entry.update(state=outcome, usage=usage)
        put(self.path, self.state, replace=True)


class APIBackend:
    def __init__(self, config, folder, *, key=None, counter=None, transport=http_post,
                 clock=time.time, sleep=time.sleep, budget=None):
        self.config = config
        self.counter = counter or ReferenceCounter(config['tokenizer_path'])
        self.transport, self.clock, self.sleep = transport, clock, sleep
        self.key = key if key is not None else read_credential()
        self.budget = budget if budget is not None else Budget(Path(folder) / 'api-budget.json', config, clock=clock)
        self.last_generation_metadata = {}

    def before_source(self):
        # Leave remaining sources pending instead of starting one without its full call allowance.
        self.budget.check(BUDGET['max_planner_attempts'] + BUDGET['max_skill_steps'])

    def __call__(self, messages):
        self.last_generation_metadata = {'input_tokens': None, 'output_tokens': None,
                                         'truncation_at_budget': False, 'error': None}
        entry = None
        try:
            if not isinstance(messages, list) or not messages or any(
                    set(m) != {'role', 'content'} or m['role'] not in ('system', 'user', 'assistant')
                    or not isinstance(m['content'], str) for m in messages):
                raise APIError('text_message_contract')
            tokens = self.counter(messages)
            self.last_generation_metadata['reference_input_tokens'] = tokens
            if tokens + GENERATION['max_new_tokens'] > GENERATION['context_cap']:
                raise APIError('reference_context_budget_exceeded')
            body = {'model': MODEL, 'messages': deepcopy(messages), 'temperature': 0.,
                    'max_tokens': GENERATION['max_new_tokens'],
                    'stream': False, 'enable_thinking': False}
            attempts = self.budget.state['attempts']
            if attempts:
                remaining = (attempts[-1]['sent_at_unix'] + self.config['limits']['min_interval_seconds']
                             - self.clock())
                if remaining > 0:
                    self.sleep(remaining)
            reserve_started = time.monotonic()
            entry = self.budget.reserve(body)
            self.last_generation_metadata['rate_and_budget_wait_seconds'] = time.monotonic() - reserve_started
            self.last_generation_metadata['api_attempt'] = entry['number']
            transport_started = time.monotonic()
            try:
                response = self.transport(self.config['base_url'] + '/chat/completions', self.key,
                                          body, self.config['limits']['request_timeout_seconds'])
            finally:
                self.last_generation_metadata['transport_seconds'] = time.monotonic() - transport_started
            usage = response.get('usage', {})
            if response.get('model') != MODEL or not isinstance(usage, dict):
                raise APIError('response_model_or_usage_mismatch')
            inp, out = usage.get('prompt_tokens'), usage.get('completion_tokens')
            if type(inp) is not int or type(out) is not int or inp < 1 or out < 0:
                raise APIError('missing_usage_counts')
            self.last_generation_metadata.update(input_tokens=inp, output_tokens=out,
                provider_model=response['model'], provider_precision=self.config['provider_precision'],
                tokenizer_comparison='local_reference_only_provider_template_not_verified')
            for key in ('id', 'system_fingerprint'):
                value = response.get(key)
                if isinstance(value, str) and re.fullmatch(r'[A-Za-z0-9_.:-]{1,256}', value) and self.key not in value:
                    self.last_generation_metadata['provider_' + key] = value
            choices = response.get('choices')
            if not isinstance(choices, list) or len(choices) != 1:
                raise APIError('response_choice_contract')
            message = choices[0].get('message', {})
            content, reason = message.get('content'), choices[0].get('finish_reason')
            self.last_generation_metadata.update(finish_reason=reason,
                truncation_at_budget=reason == 'length' or out >= GENERATION['max_new_tokens'])
            if inp + GENERATION['max_new_tokens'] > GENERATION['context_cap'] or out > GENERATION['max_new_tokens']:
                raise APIError('provider_token_budget_exceeded')
            if (not isinstance(content, str) or not content.strip() or reason not in ('stop', 'length')
                    or message.get('reasoning') or message.get('reasoning_content') or '<think>' in content
                    or self.key in content):
                raise APIError('response_content_or_thinking_contract')
            self.budget.finish(entry, 'response_received', {'prompt_tokens': inp, 'completion_tokens': out})
            return content
        except Exception as exc:
            # The runtime stores str(exc); do not let HTTP errors, headers or response bodies escape.
            code = str(exc) if isinstance(exc, APIError) else 'api_contract_error'
            self.last_generation_metadata['error'] = code
            if entry is not None:
                self.budget.finish(entry, code)
            raise APIError(code) from None
