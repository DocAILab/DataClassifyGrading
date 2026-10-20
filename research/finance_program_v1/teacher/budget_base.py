"""Append-only cross-process campaign accounting; no secret or response bodies."""
# 追加日志保存跨进程请求与用量账本；线程锁和文件锁共同保护申领、结算及停止状态。
from contextlib import contextmanager
import json
import os
from pathlib import Path
import threading

_LOCKS = {}
_LOCKS_GUARD = threading.Lock()


class PersistentBudget:
    def __init__(self, path, *, requests=1200, tokens=10000000, failures=8):
        self.path = Path(path).resolve(); self.path.mkdir(parents=True, exist_ok=True)
        self.limits = {'request_cap': requests, 'known_token_cap': tokens, 'failed_attempt_cap': failures}
        assert all(type(n) is int and n > 0 for n in self.limits.values())
        with _LOCKS_GUARD:self.lock = _LOCKS.setdefault(str(self.path), threading.RLock())
        self.local = threading.local()

    @contextmanager
    def _locked(self):
        with self.lock:
            lockpath = self.path / 'budget.lock'
            try:
                with lockpath.open('xb') as stream:stream.write(b'0')
            except FileExistsError:pass
            with lockpath.open('r+b') as stream:
                if os.name == 'nt':
                    import msvcrt
                    msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
                try:yield
                finally:
                    stream.seek(0)
                    if os.name == 'nt':msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                    else:fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    def _append(self, event):
        with (self.path / 'events.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(event, sort_keys=True, separators=(',', ':')) + '\n')
            stream.flush(); os.fsync(stream.fileno())

    def _state(self):
        journal = self.path / 'events.jsonl'
        if not journal.exists():self._append({'type': 'limits', **self.limits})
        events = [json.loads(s) for s in journal.read_text(encoding='utf-8').splitlines()]
        assert events[0] == {'type': 'limits', **self.limits}, 'campaign limits changed'
        claims = set(); done = set(); tokens = failures = unknown = 0; stopped = False
        for event in events[1:]:
            if event['type'] == 'claim':
                assert event['id'] == len(claims) + 1; claims.add(event['id'])
            elif event['type'] == 'result':
                assert event['id'] in claims - done; done.add(event['id'])
                tokens += event['known_total_tokens']; failures += not event['complete']; unknown += event['unknown_usage']
            elif event['type'] == 'stop':stopped = True
            else:raise ValueError('unknown campaign event')
        return {'requests': len(claims), 'known_total_tokens': tokens, 'failed_attempts': failures,
                'unknown_usage_attempts': unknown, 'inflight_requests': len(claims - done),
                'stopped': stopped or len(claims) >= self.limits['request_cap'] or tokens >= self.limits['known_token_cap'] or failures >= self.limits['failed_attempt_cap'],
                **self.limits}

    def claim(self):
        with self._locked():
            state = self._state()
            if state['stopped']:raise RuntimeError('teacher safety budget halted')
            assert getattr(self.local, 'pending', None) is None
            self.local.pending = state['requests'] + 1
            # 请求发出前持久化占位；进程中断后该次申领仍保留，避免恢复运行时漏记。
            self._append({'type': 'claim', 'id': self.local.pending, 'pid': os.getpid()})

    def record(self, result):
        usage = result.get('usage'); tokens = usage['total_tokens'] if usage is not None else 0
        assert type(tokens) is int and tokens >= 0 and type(result['complete']) is bool
        with self._locked():
            self._state(); claim = getattr(self.local, 'pending', None); assert claim is not None
            self._append({'type': 'result', 'id': claim, 'complete': result['complete'],
                          'known_total_tokens': tokens, 'unknown_usage': usage is None})
            self.local.pending = None

    def snapshot(self):
        with self._locked():return self._state()

    def stop(self):
        with self._locked():self._state(); self._append({'type': 'stop', 'pid': os.getpid()})


class RunBudget:
    def __init__(self, path, *, requests, tokens, failures):
        self.global_budget = PersistentBudget(path)
        self.request_cap = requests; self.token_cap = tokens; self.failure_cap = failures
        self.requests = 0; self.tokens = 0; self.failures = 0; self.unknown = 0; self.stopped = False
        self.lock = threading.RLock()

    def claim(self):
        with self.lock:
            if self.stopped or self.requests >= self.request_cap or self.tokens >= self.token_cap or self.failures >= self.failure_cap:
                self.stopped = True; raise RuntimeError('teacher safety budget halted')
            self.global_budget.claim(); self.requests += 1

    def record(self, result):
        self.global_budget.record(result)
        with self.lock:
            usage = result.get('usage')
            if usage is None:self.unknown += 1
            else:self.tokens += usage['total_tokens']
            self.failures += not result['complete']

    def snapshot(self):
        with self.lock:
            global_state = self.global_budget.snapshot()
            return {'requests': self.requests, 'known_total_tokens': self.tokens, 'failed_attempts': self.failures,
                    'unknown_usage_attempts': self.unknown, 'stopped': self.stopped or global_state['stopped'],
                    'request_cap': self.request_cap, 'known_token_cap': self.token_cap,
                    'failed_attempt_cap': self.failure_cap, 'campaign': global_state}

    def stop(self):
        with self.lock:self.stopped = True
