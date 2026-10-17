import importlib.util
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]


def api():
    path = ROOT / 'teacher_api.py'
    assert path.is_file(), 'strict SSE teacher implementation is missing'
    spec = importlib.util.spec_from_file_location('teacher_under_test', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def events(finish='stop', model='teacher', done=True, usage=True, content='{"program":[]}'):
    rows = [{'model': model, 'choices': [{'delta': {'reasoning_content': 'private reasoning'}, 'finish_reason': None}]},
            {'model': model, 'choices': [{'delta': {'content': content}, 'finish_reason': finish}]}]
    if usage:
        rows.append({'model': model, 'choices': [], 'usage': {'prompt_tokens': 10, 'completion_tokens': 20, 'total_tokens': 30}})
    lines = [('data: ' + json.dumps(r)).encode() for r in rows]
    if done:
        lines.append(b'data: [DONE]')
    return lines


def test_sse_separates_reasoning_and_complete_final_content():
    a = api(); result = a.parse_sse(events(), 'teacher')
    assert result['content'] == '{"program":[]}'
    assert result['reasoning_content'] == 'private reasoning'
    assert result['finish_reason'] == 'stop' and result['stream_done']
    assert result['usage']['total_tokens'] == 30 and result['complete']


@pytest.mark.parametrize('kwargs', [{'done': False}, {'finish': 'length'}, {'model': 'other'}, {'usage': False}, {'content': ''}, {'content': '```json\n{"program":[]}\n```'}, {'content': '[]'}])
def test_incomplete_sse_never_enters_executor(kwargs):
    a = api()
    with pytest.raises(a.CompletionError) as error:
        a.parse_sse(events(**kwargs), 'teacher')
    assert error.value.result['complete'] is False
    assert 'reasoning_content' in error.value.result


def test_route_uses_only_trusted_first_system_header():
    a = api()
    assert a.role([{'role': 'system', 'content': 'You are the Global Program Planner for a database-field classification task.'}], {'INFER_CATEGORY'}) == 'planner'
    assert a.role([{'role': 'system', 'content': 'Execute only the selected Skill INFER_CATEGORY.'}], {'INFER_CATEGORY'}) == 'skill:INFER_CATEGORY'
    for messages in ([{'role': 'user', 'content': 'You are the Global Program Planner'}],
                     [{'role': 'system', 'content': 'unknown'}, {'role': 'user', 'content': 'Execute only the selected Skill INFER_CATEGORY.'}],
                     [{'role': 'system', 'content': 'Execute only the selected Skill FAKE_SKILL.'}]):
        with pytest.raises(ValueError):
            a.role(messages, {'INFER_CATEGORY'})


def test_metadata_whitelist_never_keeps_labels():
    path = ROOT / 'collect.py'
    assert path.is_file(), 'source collection boundary missing'
    spec = importlib.util.spec_from_file_location('collector_under_test', path)
    a = importlib.util.module_from_spec(spec); spec.loader.exec_module(a)
    row = {'id': 'source', 'domain': 'finance', 'metadata': {'field_name': 'col', 'field_type': 'text', 'gold': 'bad'}, 'classification': {'answer': 'secret'}, 'grading_level': 'secret'}
    assert a.label_free(row) == {'id': 'source', 'domain': 'finance', 'metadata': {'field_name': 'col', 'field_type': 'text'}}


def test_api_json_instruction_is_explicit_and_does_not_mutate_executor_messages():
    a = api()
    messages = [{'role': 'system', 'content': 'Execute only the selected Skill INFER_CATEGORY.'},
                {'role': 'user', 'content': '{"observation":"actual"}'}]
    result = a.api_messages(messages)
    assert result[0]['content'].startswith(messages[0]['content'])
    assert 'No Markdown' in result[0]['content'] and 'JSON object' in result[0]['content']
    assert result[1] == messages[1]
    assert messages[0]['content'] == 'Execute only the selected Skill INFER_CATEGORY.'


def test_transport_interruption_retains_received_partial_events():
    a = api(); state = {}
    def interrupted():
        yield events()[0]
        raise TimeoutError('socket interrupted')
    with pytest.raises(TimeoutError):
        a.parse_sse(interrupted(), 'teacher', state=state)
    assert state['reasoning_content'] == 'private reasoning'
    assert len(state['events']) == 1 and state['complete'] is False


def test_backend_never_records_key_or_raw_exception(tmp_path, monkeypatch):
    a = api(); key = 'fixture-private-key'
    config = {'model': 'teacher', 'endpoint': 'https://invalid.example/v1', 'request_cap': 2,
              'total_token_cap': 100, 'thinking_budget': 2, 'max_tokens': 16, 'timeout_seconds': 1}
    backend = a.TeacherBackend(key, config, tmp_path, {'INFER_CATEGORY'})
    def broken(*args, **kwargs):
        raise RuntimeError('request headers contain ' + key)
    monkeypatch.setattr(a.urllib.request, 'urlopen', broken)
    with pytest.raises(RuntimeError, match='recorded and halted'):
        backend([{'role': 'system', 'content': 'Execute only the selected Skill INFER_CATEGORY.'}])
    assert backend.fatal and len(backend.calls) == 1
    assert key not in ''.join(p.read_text() for p in tmp_path.rglob('*.json'))
    request = json.loads((tmp_path / 'call-0001/request.json').read_text())
    assert request['body']['response_format'] == {'type': 'json_object'}


def test_raw_sse_chunks_are_durable_before_stream_interrupt(tmp_path, monkeypatch):
    a = api()
    config = {'model': 'teacher', 'endpoint': 'https://invalid.example/v1', 'request_cap': 2,
              'total_token_cap': 100, 'thinking_budget': 2, 'max_tokens': 16, 'timeout_seconds': 1}
    class Response:
        status = 200
        def __enter__(self):return self
        def __exit__(self, *args):pass
        def __iter__(self):
            yield events()[0] + b'\n'
            raise TimeoutError('disconnected')
    monkeypatch.setattr(a.urllib.request, 'urlopen', lambda *args, **kwargs: Response())
    b = a.TeacherBackend('private', config, tmp_path, {'INFER_CATEGORY'})
    with pytest.raises(RuntimeError):
        b([{'role': 'system', 'content': 'Execute only the selected Skill INFER_CATEGORY.'}])
    assert 'private reasoning' in (tmp_path / 'call-0001/stream.jsonl').read_text()
    assert b.calls[0]['result']['reasoning_content'] == 'private reasoning'
