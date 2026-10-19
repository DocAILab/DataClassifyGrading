"""Offline reference-tokenizer check; --live explicitly performs two synthetic API requests."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

from api_backend import APIBackend, ReferenceCounter, api_identity, load_config
from artifacts import GENERATION, put, require_private_output


def synthetic_messages(counter):
    short = [{'role': 'system', 'content': 'Return only the JSON object {"ok": true}.'},
             {'role': 'user', 'content': 'This is a synthetic connectivity check. Return the requested JSON.'}]
    low, high = 1, GENERATION['context_cap']
    target = GENERATION['context_cap'] - GENERATION['max_new_tokens'] - 64
    while low < high:
        mid = (low + high + 1) // 2
        probe = [short[0], {'role': 'user', 'content': 'padding ' * mid + '\nReturn the requested JSON.'}]
        if counter(probe) <= target:
            low = mid
        else:
            high = mid - 1
    long = [short[0], {'role': 'user', 'content': 'padding ' * low + '\nReturn the requested JSON.'}]
    if not 32000 <= counter(long) <= target:
        raise ValueError('synthetic boundary prompt did not reach the target context')
    return [short, long]


def check(config_path, output, *, live=False):
    config = load_config(config_path)
    counter = ReferenceCounter(config['tokenizer_path'])
    prompts = synthetic_messages(counter)
    output = Path(output)
    require_private_output(output)
    output.mkdir(parents=True, exist_ok=False)
    report = {'scope': 'synthetic only; no vehicle inputs or gold', 'live': live,
              'service': api_identity(config), 'calls': [], 'http_requests': 0}
    put(output / 'check-started.json', {'live': live, 'service': api_identity(config)})
    backend = None
    try:
        if live:
            backend = APIBackend(config, output / 'backend', counter=counter)
        for messages in prompts:
            item = {'reference_input_tokens': counter(messages), 'capacity_checked_locally': True}
            if backend:
                text = backend(messages)
                item.update(generation=deepcopy(backend.last_generation_metadata), response_json_ok=json.loads(text) == {'ok': True})
                if not item['response_json_ok']:
                    raise ValueError('synthetic JSON instruction failed')
            report['calls'].append(item)
        report['passed'] = True
    except Exception as exc:
        report.update(passed=False, error_type=type(exc).__name__)
    finally:
        if backend:
            report['http_requests'] = len(backend.budget.state['attempts'])
        put(output / 'check.json', report)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--live', action='store_true')
    args = p.parse_args()
    result = check(args.config, args.output, live=args.live)
    print(json.dumps({'passed': result['passed'], 'live': result['live'], 'http_requests': result['http_requests']}))
    raise SystemExit(0 if result['passed'] else 2)
