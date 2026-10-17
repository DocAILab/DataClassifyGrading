import importlib.util
from pathlib import Path
import threading
import pytest

ROOT = Path(__file__).resolve().parents[1]


def api():
    path = ROOT / 'retry_backend.py'
    assert path.exists(), 'bounded recorded retry implementation missing'
    spec = importlib.util.spec_from_file_location('retry_under_test', path)
    a = importlib.util.module_from_spec(spec); spec.loader.exec_module(a)
    return a


@pytest.mark.parametrize('result,expected', [
    ({'complete': False, 'error_type': 'IncompleteCompletion', 'models': ['teacher'], 'finish_reason': None, 'stream_done': False}, True),
    ({'complete': False, 'error_type': 'IncompleteCompletion', 'models': ['wrong'], 'finish_reason': None}, False),
    ({'complete': False, 'error_type': 'IncompleteCompletion', 'models': ['teacher'], 'finish_reason': 'length'}, False),
    ({'complete': False, 'error_type': 'IncompleteCompletion', 'models': ['teacher'], 'finish_reason': 'stop', 'stream_done': True}, False),
    ({'complete': False, 'error_type': 'HTTPError', 'http_status': 403}, False),
    ({'complete': False, 'error_type': 'HTTPError', 'http_status': 503}, True),
    ({'complete': False, 'error_type': 'TimeoutError'}, True),
    ({'complete': True}, False),
])
def test_retry_is_only_transient_and_not_semantic_repair(result, expected):
    assert api().retryable(result, 'teacher') is expected


def test_retry_preserves_attempts_and_returns_only_complete_final(tmp_path):
    a = api()
    class Inner:
        calls = []; fatal = False; total_tokens = 0
        def __call__(self, messages):
            failed = not self.calls
            result = {'complete': not failed, 'content': '' if failed else '{"ok":true}', 'models': ['teacher'],
                      'error_type': 'IncompleteCompletion' if failed else None, 'finish_reason': None if failed else 'stop',
                      'stream_done': not failed, 'usage': None if failed else {'total_tokens': 10}}
            self.calls.append({'request': {'executor_messages': messages}, 'result': result})
            self.fatal = failed
            if failed:raise RuntimeError('recorded incomplete')
            return result['content']
    inner = Inner()
    budget = a.Budget(10, 1000, 5)
    wrapper = a.RetryBackend(inner, 'teacher', tmp_path, budget, retries=1, backoff=0)
    assert wrapper([{'role': 'system', 'content': 'same request'}]) == '{"ok":true}'
    assert len(inner.calls) == 2 and inner.calls[0]['result']['complete'] is False
    assert not wrapper.fatal and budget.snapshot()['requests'] == 2
    assert budget.snapshot()['unknown_usage_attempts'] == 1
    assert (tmp_path / 'logical-0001.json').exists()


def test_shared_budget_is_atomic_under_parallel_sources():
    a = api(); b = a.Budget(5, 1000, 5); accepted = []
    def claim():
        try:b.claim(); accepted.append(True)
        except RuntimeError:pass
    threads = [threading.Thread(target=claim) for _ in range(30)]
    for t in threads:t.start()
    for t in threads:t.join()
    assert len(accepted) == 5 and b.snapshot()['requests'] == 5
