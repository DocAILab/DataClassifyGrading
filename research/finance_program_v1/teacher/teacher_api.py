"""Strict, bounded, credential-free-on-disk Bailian SSE backend."""
# 一次 HTTP 请求对应一组不可覆盖的请求与流式响应记录，完整性门禁决定响应能否交给执行器。
import hashlib
from copy import deepcopy
import json
from pathlib import Path
import re
import time
import urllib.error
import urllib.request


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


class CompletionError(RuntimeError):
    def __init__(self, result):
        self.result = result
        super().__init__('teacher response failed complete-model/stop/DONE/usage gate')


def parse_sse(lines, model, *, state=None):
    result = state if state is not None else {}
    result.update({'content': '', 'reasoning_content': '', 'models': [], 'finish_reason': None,
                   'stream_done': False, 'usage': None, 'events': [], 'complete': False})
    for raw in lines:
        line = raw.decode('utf-8').strip() if isinstance(raw, bytes) else raw.strip()
        if not line.startswith('data:'):
            continue
        data = line[5:].strip()
        if data == '[DONE]':
            result['stream_done'] = True
            break
        try:
            obj = json.loads(data)
        except (ValueError, TypeError):
            result['malformed_event'] = True
            raise CompletionError(result) from None
        result['events'].append(obj)
        if obj.get('model') and obj['model'] not in result['models']:
            result['models'].append(obj['model'])
        if obj.get('usage'):
            result['usage'] = obj['usage']
        for choice in obj.get('choices', []):
            if choice.get('index', 0) != 0:
                raise CompletionError(result)
            delta = choice.get('delta', {})
            result['content'] += delta.get('content') or ''
            result['reasoning_content'] += delta.get('reasoning_content') or ''
            if choice.get('finish_reason'):
                result['finish_reason'] = choice['finish_reason']
    usage = result['usage']
    try:
        json_object = isinstance(json.loads(result['content']), dict)
    except (ValueError, TypeError):
        json_object = False
    # JSON 可解析仍不足以接收：还需正常停止、流结束、模型身份及完整用量同时成立。
    result['complete'] = bool(result['stream_done'] and result['finish_reason'] == 'stop' and
        result['models'] == [model] and json_object and isinstance(usage, dict) and
        all(type(usage.get(k)) is int and usage[k] >= 0 for k in ('prompt_tokens', 'completion_tokens', 'total_tokens')) and
        usage['total_tokens'] == usage['prompt_tokens'] + usage['completion_tokens'])
    if not result['complete']:
        raise CompletionError(result)
    return result


def role(messages, allowed_skills):
    if not messages or messages[0].get('role') != 'system':
        raise ValueError('trusted first system message required')
    head = messages[0]['content'].splitlines()[0]
    if head == 'You are the Global Program Planner for a database-field classification task.':
        return 'planner'
    match = re.fullmatch(r'Execute only the selected Skill ([A-Z_]+)\.', head)
    if match and match[1] in allowed_skills:
        return 'skill:' + match[1]
    raise ValueError('unknown trusted model role')


def api_messages(messages):
    from policy import apply_policy
    value = apply_policy(messages)
    value[0]['content'] += ('\nTransport output contract: The final content must be exactly one raw JSON object. '
        'No Markdown, no code fences, no prose before or after the JSON object. '
        'Put any reasoning only in the separate reasoning channel. Preserve all original task contracts and budgets.')
    return value


class TeacherBackend:
    def __init__(self, key, config, call_dir, allowed_skills):
        self._key = key
        self.config = config
        self.call_dir = Path(call_dir)
        self.allowed_skills = set(allowed_skills)
        self.calls = []
        self.total_tokens = 0
        self.fatal = False

    def __call__(self, messages):
        if self.fatal:
            raise RuntimeError('teacher system halted; see recorded call failure')
        if len(self.calls) >= self.config['request_cap'] or self.total_tokens >= self.config['total_token_cap']:
            self.fatal = True
            raise RuntimeError('teacher request/token safety cap reached')
        selected_role = role(messages, self.allowed_skills)
        body = {'model': self.config['model'], 'messages': api_messages(messages), 'stream': True,
                'stream_options': {'include_usage': True}, 'temperature': 0,
                'thinking_budget': self.config['thinking_budget'], 'max_tokens': self.config['max_tokens'],
                'response_format': {'type': 'json_object'}}
        folder = self.call_dir / ('call-%04d' % (len(self.calls) + 1))
        folder.mkdir(parents=True, exist_ok=False)
        request_record = {'role': selected_role, 'body': body, 'body_sha256': digest(body),
                          'executor_messages': deepcopy(messages),
                          'endpoint': self.config['endpoint'], 'credential_recorded': False}
        save(folder / 'request.json', request_record)
        result = {'complete': False, 'content': '', 'reasoning_content': '', 'usage': None,
                  'finish_reason': None, 'stream_done': False, 'models': [], 'events': []}
        started = time.monotonic()
        try:
            # Headers remain in memory; request records deliberately contain no headers.
            request = urllib.request.Request(self.config['endpoint'] + '/chat/completions',
                data=json.dumps(body, ensure_ascii=False).encode(),
                headers={'Authorization': 'Bearer ' + self._key, 'Content-Type': 'application/json'}, method='POST')
            with urllib.request.urlopen(request, timeout=self.config['timeout_seconds']) as response:
                def bounded_lines():
                    with (folder / 'stream.jsonl').open('x', encoding='utf-8') as stream:
                        for raw in response:
                            stream.write(json.dumps({'line': raw.decode('utf-8')}, ensure_ascii=False) + '\n')
                            stream.flush()
                            if time.monotonic() - started > self.config['timeout_seconds']:
                                raise TimeoutError('teacher wall deadline')
                            yield raw
                result = parse_sse(bounded_lines(), self.config['model'], state=result)
                result['http_status'] = response.status
        except CompletionError as exc:
            result = exc.result
            result['error_type'] = 'IncompleteCompletion'
            self.fatal = True
        except urllib.error.HTTPError as exc:
            result['error_type'] = 'HTTPError'; result['http_status'] = exc.code
            self.fatal = True
        except Exception as exc:
            # Never log exception text, request repr, or response body containing credentials.
            result['error_type'] = type(exc).__name__
            self.fatal = True
        finally:
            result['seconds'] = time.monotonic() - started
            result['role'] = selected_role
            result['response_sha256'] = digest({'content': result['content'], 'reasoning_content': result['reasoning_content']})
            self.total_tokens += (result.get('usage') or {}).get('total_tokens', 0)
            save(folder / 'result.json', result)
            self.calls.append({'request': request_record, 'result': result})
        if not result['complete']:
            raise RuntimeError('teacher response incomplete; recorded and halted')
        return result['content']
