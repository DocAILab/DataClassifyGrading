"""Retry-aware recorded-I/O replay helpers; no private post-hoc gold audit."""
import json
from pathlib import Path
from teacher_api import api_messages, digest, parse_sse, role, save
from retry_backend import retryable


def validate_record(record, messages, allowed, allow_fixture=False):
    q, x = record['request'], record['result']
    assert q['body_sha256'] == digest(q['body'])
    assert q['executor_messages'] == messages
    assert q['role'] == role(messages, allowed)
    if x.get('fixture'):
        assert allow_fixture
        return
    assert q['body']['messages'] == api_messages(messages)
    assert q['body']['model'] == 'qwen3.8-2.4t-a95b'
    assert q['body']['response_format'] == {'type': 'json_object'}
    assert q['credential_recorded'] is False and 'headers' not in q
    if x['complete']:
        lines = ['data: ' + json.dumps(e) for e in x['events']]
        if x['stream_done']:lines.append('data: [DONE]')
        rebuilt = parse_sse(lines, q['body']['model'])
        for k in ('content', 'reasoning_content', 'usage', 'models', 'finish_reason', 'stream_done', 'complete'):
            assert rebuilt[k] == x[k]
    assert x['response_sha256'] == digest({'content': x['content'], 'reasoning_content': x['reasoning_content']})


def replay_retry_one(study, row, saved, records, logical, allowed, *, allow_fixture=False):
    from program_loop import build_program_controller
    class Recorded:
        def __init__(self):self.position = 0; self.http_position = 0; self.faults = []; self.halted = False
        def __call__(self, messages):
            if self.halted:
                raise RuntimeError('teacher system halted; see recorded call failure')
            try:
                assert self.position < len(logical)
                group = logical[self.position]; self.position += 1
                assert group['executor_messages_sha256'] == digest(messages)
                assert group['call_start'] == self.http_position
                attempts = records[group['call_start']:group['call_end']]
                assert 0 <= len(attempts) <= 2
                for n, record in enumerate(attempts):
                    validate_record(record, messages, allowed, allow_fixture)
                    if n < len(attempts) - 1:
                        assert retryable(record['result'], record['request']['body']['model'])
                self.http_position = group['call_end']
                if group['complete']:
                    assert attempts and attempts[-1]['result']['complete'] is True and group['error'] is None
                    return attempts[-1]['result']['content']
                assert group['error'] in ('teacher response incomplete; recorded and halted', 'teacher safety budget halted', 'teacher source backend failed before receipt')
                assert not attempts or not attempts[-1]['result']['complete']
            except Exception:
                self.faults.append('logical/HTTP recorded boundary mismatch'); raise
            self.halted = True
            raise RuntimeError(group['error'])
    backend = Recorded()
    c = build_program_controller(backend, study['registry'], study['rows'], study['budget'], skill_pool='F66')
    actual = c.run({'source_id': row['id'], 'domain': row['domain'], 'field_name': row['metadata']['field_name']}).to_dict()
    assert not backend.faults and backend.position == len(logical) and backend.http_position == len(records)
    assert actual == saved, 'full original trace differs'
    return {'full_trace_exact': True, 'all_calls_consumed': True, 'logical_calls': len(logical),
            'api_attempts': len(records), 'failed_api_attempts': sum(not r['result']['complete'] for r in records),
            'terminated': actual['terminated'], 'stop_reason': actual['stop_reason']}


def read_records(folder):
    from collect import read
    paths = sorted(folder.glob('call-*'))
    assert [p.name for p in paths] == ['call-%04d' % n for n in range(1, len(paths) + 1)]
    records = []
    for p in paths:
        q, x = read(p / 'request.json'), read(p / 'result.json')
        if x['complete']:
            raw = [json.loads(line)['line'] for line in (p / 'stream.jsonl').read_text(encoding='utf-8').splitlines()]
            result = parse_sse(raw, q['body']['model'])
            for k in ('content', 'reasoning_content', 'usage', 'models', 'finish_reason', 'stream_done', 'events'):
                assert result[k] == x[k]
        records.append({'request': q, 'result': x})
    return records



# The old audit_all aggregator required private historical runs and gold.
# Only the recorded-I/O replay helpers used by detached.py are published here.
