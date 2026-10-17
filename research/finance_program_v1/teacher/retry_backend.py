"""One recorded transport retry, shared atomic request/usage safety counters."""
import threading
import time
from teacher_api import digest, save


def retryable(result, model):
    if result.get('complete') or result.get('malformed_event'):
        return False
    if result.get('models') not in (None, [], [model]):
        return False
    kind = result.get('error_type')
    if kind == 'HTTPError':
        return result.get('http_status') in (429, 500, 502, 503, 504)
    if kind in ('TimeoutError', 'URLError', 'RemoteDisconnected', 'ConnectionResetError', 'IncompleteRead'):
        return True
    return (kind == 'IncompleteCompletion' and result.get('finish_reason') is None and
            result.get('stream_done') is not True)


class Budget:
    def __init__(self, requests, tokens, failures):
        self.request_cap = requests; self.token_cap = tokens; self.failure_cap = failures
        self.requests = 0; self.tokens = 0; self.failures = 0; self.unknown = 0
        self.lock = threading.Lock(); self.stopped = False

    def claim(self):
        with self.lock:
            if self.stopped or self.requests >= self.request_cap or self.tokens >= self.token_cap or self.failures >= self.failure_cap:
                self.stopped = True
                raise RuntimeError('teacher safety budget halted')
            self.requests += 1

    def record(self, result):
        with self.lock:
            usage = result.get('usage')
            if usage is None:
                self.unknown += 1
            else:
                self.tokens += usage['total_tokens']
            self.failures += not result.get('complete', False)

    def stop(self):
        with self.lock:self.stopped = True

    def snapshot(self):
        with self.lock:
            return {'requests': self.requests, 'known_total_tokens': self.tokens, 'failed_attempts': self.failures,
                    'unknown_usage_attempts': self.unknown, 'stopped': self.stopped,
                    'request_cap': self.request_cap, 'known_token_cap': self.token_cap, 'failed_attempt_cap': self.failure_cap}


class RetryBackend:
    def __init__(self, inner, model, logical_dir, budget, *, retries=1, backoff=3):
        if retries != 1:
            raise ValueError('exactly one retry maximum is frozen')
        self.inner = inner; self.model = model; self.logical_dir = logical_dir; self.budget = budget
        self.retries = retries; self.backoff = backoff; self.logical = []; self.fatal = False

    @property
    def calls(self):return self.inner.calls

    def __call__(self, messages):
        if self.fatal:
            raise RuntimeError('teacher system halted; see recorded call failure')
        start = len(self.calls)
        proof = {'executor_messages_sha256': digest(messages), 'call_start': start, 'call_end': start,
                 'complete': False, 'error': None, 'retry_policy': 'one transport-only retry; identical messages'}
        try:
            for index in range(self.retries + 1):
                try:
                    self.budget.claim()
                except RuntimeError:
                    self.fatal = True; proof['error'] = 'teacher safety budget halted'
                    raise RuntimeError(proof['error']) from None
                before = len(self.calls)
                try:
                    value = self.inner(messages)
                except RuntimeError:
                    if len(self.calls) != before + 1:
                        self.fatal = True; proof['error'] = 'teacher source backend failed before receipt'
                        raise RuntimeError(proof['error']) from None
                    result = self.calls[-1]['result']; self.budget.record(result)
                    transient = retryable(result, self.model)
                    if transient and index < self.retries:
                        self.inner.fatal = False
                        time.sleep(self.backoff)
                        continue
                    self.fatal = True
                    if not transient:
                        self.budget.stop()
                    proof['error'] = 'teacher response incomplete; recorded and halted'
                    raise RuntimeError(proof['error']) from None
                else:
                    self.budget.record(self.calls[-1]['result']); proof['complete'] = True
                    return value
        finally:
            proof['call_end'] = len(self.calls)
            self.logical.append(proof)
            save(self.logical_dir / ('logical-%04d.json' % len(self.logical)), proof)
