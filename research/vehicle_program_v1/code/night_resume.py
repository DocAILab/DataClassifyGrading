"""Fail-closed reconciliation of night HTTP evidence before any resumed network access."""
import math
from pathlib import Path

from api_backend import MODEL, api_identity
from artifacts import GENERATION, fingerprint, read, sha
from runner import environment, read_receipts

TRANSIENT = {'transport_or_json_error', 'http_408', 'http_429', 'http_500', 'http_502', 'http_503', 'http_504'}


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def effective_cooldown(ledger, policy):
    """Conservatively reconstruct admission delay, including failures drained after stop."""
    deadline = ledger.get('cooldown_until_unix', 0.)
    if not finite_number(deadline) or deadline < 0:
        raise RuntimeError('resume_cooldown_invalid')
    for attempt in ledger['attempts']:
        if attempt['state'] in TRANSIENT:
            finished = attempt.get('finished_at_unix')
            if not finite_number(finished) or finished <= 0:
                raise RuntimeError('resume_reservation_invalid')
            delay = policy['rate_limit_cooldown_seconds'] if attempt['state'] == 'http_429' else policy['transient_cooldown_seconds']
            deadline = max(deadline, finished + delay)
    return deadline


def validate_resume_ledger(plan, study, seed, config):
    output = Path(plan['output'])
    path = output / 'backend/api-budget.json'
    if not path.is_file():
        raise RuntimeError('resume_budget_missing_requires_review')
    ledger = read(path)
    attempts = ledger['attempts']
    if any(a['state'] not in TRANSIENT | {'response_received'} for a in attempts):
        raise RuntimeError('uncertain_or_fatal_attempt_requires_review')
    if not (output / 'run.json').is_file():
        raise RuntimeError('resume_run_identity_missing')
    identity = read(output / 'run.json')
    manifest = read(Path(plan['package']) / 'manifest.json')
    start = identity.get('wall_started_at_unix')
    if (identity.get('schema') != 4 or not finite_number(start) or start <= 0
            or ledger.get('started_at_unix') != start
            or identity.get('night_plan_sha256') != fingerprint(plan)
            or identity.get('study_sha256') != manifest['study_sha256']
            or identity.get('code_sha256') != fingerprint(manifest['code'])
            or identity.get('expected_ids') != study['expected_ids']
            or identity.get('environment') != environment()
            or identity.get('api_service') != api_identity(config)
            or ledger.get('config_sha256') != fingerprint({'api': api_identity(config), 'scheduler': plan['scheduler']})):
        raise RuntimeError('resume_budget_identity_or_wall_origin_differs')
    if [a['number'] for a in attempts] != list(range(1, len(attempts) + 1)):
        raise RuntimeError('resume_budget_sequence_differs')
    effective_cooldown(ledger, plan['policy'])
    price = config['pricing_cny_per_million']
    upper = ((GENERATION['context_cap'] - GENERATION['max_new_tokens']) * price['input']
             + GENERATION['max_new_tokens'] * price['output']) / 1e6
    receipts = read_receipts(output, study['expected_ids'])
    seen = set()
    for position, ident in enumerate(study['expected_ids'], 1):
        folder = output / f'sources/{position:04d}'
        if ident in seed:
            if ident not in receipts or sha(folder / 'receipt.json') != sha(Path(plan['package']) / f'seed/sources/{position:04d}/receipt.json'):
                raise RuntimeError('resume_seed_receipt_differs')
            continue
        if not folder.exists():
            continue
        if ident not in receipts or receipts[ident]['status'] == 'interrupted':
            raise RuntimeError('interrupted_source_requires_review')
        for call_path in sorted((folder / 'calls').glob('*.json')):
            call = read(call_path)
            generation = call['generation']
            number = generation.get('api_attempt')
            if number is None:
                if call['error_type'] is None or call['raw_response'] is not None:
                    raise RuntimeError('resume_call_missing_attempt')
                continue  # A local pre-send rejection incurred no HTTP charge.
            if type(number) is not int or number < 1 or number > len(attempts) or number in seen:
                raise RuntimeError('resume_call_attempt_missing_or_duplicate')
            seen.add(number)
            entry = attempts[number - 1]
            body = {'model': MODEL, 'messages': call['messages'], 'temperature': 0.,
                    'max_tokens': GENERATION['max_new_tokens'], 'stream': False, 'enable_thinking': False}
            outcome = generation.get('error') if call['error_type'] else 'response_received'
            if entry['request_sha256'] != fingerprint(body) or entry['state'] != outcome:
                raise RuntimeError('resume_request_or_outcome_differs')
            if (not finite_number(entry['reserved_cny']) or abs(entry['reserved_cny'] - upper) > 1e-12
                    or not finite_number(entry['sent_at_unix']) or entry['sent_at_unix'] < start
                    or not finite_number(entry.get('finished_at_unix')) or entry['finished_at_unix'] < entry['sent_at_unix']
                    or type(entry.get('rate_tokens')) is not int or entry['rate_tokens'] <= 0):
                raise RuntimeError('resume_reservation_invalid')
            if outcome == 'response_received':
                inp, out = generation.get('input_tokens'), generation.get('output_tokens')
                if (type(inp) is not int or type(out) is not int or inp < 1 or out < 0
                        or inp + GENERATION['max_new_tokens'] > GENERATION['context_cap']
                        or out > GENERATION['max_new_tokens']):
                    raise RuntimeError('resume_usage_invalid')
                usage = {'prompt_tokens': inp, 'completion_tokens': out}
                cost = (inp * price['input'] + out * price['output']) / 1e6
                saved_cost = entry.get('usage_cost_cny')
                if (entry.get('usage') != usage or not finite_number(saved_cost)
                        or abs(saved_cost - cost) > 1e-12 or cost > upper + 1e-12):
                    raise RuntimeError('resume_usage_or_cost_differs')
            elif 'usage_cost_cny' in entry:
                raise RuntimeError('resume_uncertain_charge_released')
    if seen != set(range(1, len(attempts) + 1)):
        raise RuntimeError('resume_unmatched_budget_attempt_requires_review')
    return start
