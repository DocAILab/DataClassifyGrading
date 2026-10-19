"""Versioned, anchored limit extensions; original accounting and atomic locks reused."""
import hashlib
import json

from budget_base import PersistentBudget as OriginalBudget, RunBudget as OriginalRunBudget


class PersistentBudget(OriginalBudget):
    def _state(self):
        journal=self.path/'events.jsonl'
        if not journal.exists():self._append({'type':'limits', **self.limits})
        lines=journal.read_bytes().splitlines(keepends=True)
        assert json.loads(lines[0])=={'type':'limits', **self.limits}, 'initial limits changed'
        limits=dict(self.limits);claims=set();done=set();tokens=failures=unknown=0;stopped=False
        digest=hashlib.sha256(lines[0])
        for raw in lines[1:]:
            event=json.loads(raw);kind=event['type']
            if kind=='claim':
                assert event['id']==len(claims)+1
                claims.add(event['id'])
            elif kind=='result':
                assert event['id'] in claims-done
                assert type(event['known_total_tokens']) is int and event['known_total_tokens']>=0
                assert type(event['complete']) is bool and type(event['unknown_usage']) is bool
                done.add(event['id']);tokens+=event['known_total_tokens']
                failures+=not event['complete'];unknown+=event['unknown_usage']
            elif kind=='stop':stopped=True
            elif kind=='limit_extension':
                assert not claims-done, 'extension with inflight requests'
                assert event['prefix_sha256']==digest.hexdigest(), 'extension anchor mismatch'
                assert isinstance(event['authorization'],str) and event['authorization'].strip()
                updated={k:event[k] for k in self.limits}
                assert all(type(n) is int and n>=limits[k] for k,n in updated.items())
                assert updated['failed_attempt_cap']==limits['failed_attempt_cap'], 'failure protection cannot reset'
                assert updated!=limits, 'extension must increase a protection limit'
                limits=updated
            else:raise ValueError('unknown campaign event')
            digest.update(raw)
        return {'requests':len(claims),'known_total_tokens':tokens,'failed_attempts':failures,
                'unknown_usage_attempts':unknown,'inflight_requests':len(claims-done),
                'stopped':stopped or len(claims)>=limits['request_cap'] or tokens>=limits['known_token_cap'] or failures>=limits['failed_attempt_cap'],
                **limits}

    def extend(self, *, requests, tokens, failures, prefix_sha256, authorization):
        with self._locked():
            state=self._state()
            assert state['inflight_requests']==0
            assert hashlib.sha256((self.path/'events.jsonl').read_bytes()).hexdigest()==prefix_sha256
            updated={'request_cap':requests,'known_token_cap':tokens,'failed_attempt_cap':failures}
            assert all(type(n) is int and n>=state[k] for k,n in updated.items())
            assert failures==state['failed_attempt_cap']
            assert any(n>state[k] for k,n in updated.items())
            assert isinstance(authorization,str) and authorization.strip()
            self._append({'type':'limit_extension',**updated,'prefix_sha256':prefix_sha256,'authorization':authorization})
            return self._state()


class RunBudget(OriginalRunBudget):
    def __init__(self, path, *, requests, tokens, failures):
        super().__init__(path,requests=requests,tokens=tokens,failures=failures)
        self.global_budget=PersistentBudget(path)
