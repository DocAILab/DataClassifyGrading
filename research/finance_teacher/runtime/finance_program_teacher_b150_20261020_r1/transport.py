"""One-request SSE capture with a parent-enforced deadline; no accounting policy."""
import importlib.util
import json
import math
import multiprocessing
from pathlib import Path
import time
import urllib.request
import urllib.error
import re
import os

PARSER_PATH = Path(__file__).resolve().parent.parent / 'finance_program_teacher_balanced_20261016_r9/lib/teacher_api.py'


def _save(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.flush();os.fsync(stream.fileno())

class CaptureStorageFailure(RuntimeError):pass
class NetworkFailure(RuntimeError):pass

def _network_call(function,*args):
    try:return function(*args)
    except urllib.error.HTTPError:raise
    except (OSError,EOFError) as exc:raise NetworkFailure() from exc

def _append_raw(raw,chunk):
    try:raw.write(chunk)
    except OSError as exc:raise CaptureStorageFailure() from exc

def capture_failure(status,exitcode,timed_out):
    return bool((status or {}).get('fatal_capture') or (not timed_out and (status is None or exitcode!=0)))


def _capture_worker(folder, url, body, key):
    folder = Path(folder)
    status = {}
    try:
        request = urllib.request.Request(url, data=body if isinstance(body, bytes) else json.dumps(body,ensure_ascii=False,separators=(',', ':')).encode('utf-8'),
            headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'}, method='POST')
        # No redirects: a redirect must never turn this into a second request.
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, *args, **kwargs):
                return None
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
        try:raw=(folder / 'response.sse').open('ab', buffering=0)
        except OSError as exc:raise CaptureStorageFailure() from exc
        with _network_call(opener.open,request) as response, raw:
            status['http_status'] = response.status
            line = bytearray()
            event = []
            while True:
                chunk = _network_call(response.read1,4096)
                if not chunk:
                    break
                _append_raw(raw,chunk)
                done = False
                for byte in chunk:
                    line.append(byte)
                    if byte == 10:
                        value = bytes(line).rstrip(b'\r\n'); line.clear()
                        if value == b'':
                            if event == [b'data: [DONE]'] or event == [b'data:[DONE]']:
                                done = True
                                break
                            event.clear()
                        elif not value.startswith(b':'):
                            event.append(value)
                if done:
                    status['done_event'] = True
                    break
    except urllib.error.HTTPError as exc:
        status.update(http_status=exc.code,error_type='HTTPError')
        status['failure_class']=('auth' if exc.code in (401,403) else 'billing' if exc.code==402 else 'rate_limit' if exc.code==429 else 'transient')
        _save(folder/'http-headers.json',dict(status))
        try:data=exc.read(1048577)
        except (OSError,EOFError):data=b'';status['http_body_read_failed']=True
        status['http_body_truncated']=len(data)>1048576
        safe=data[:1048576].replace(key.encode(),b'[REDACTED_CREDENTIAL]')
        try:
            with (folder/'http-error-response.bin').open('xb') as stream:stream.write(safe)
        except OSError:
            status.update(fatal_capture=True,error_type='CaptureStorageFailure')
        code=''
        try:
            value=json.loads(safe);code=value.get('error',{}).get('code','') or value.get('code','')
        except (ValueError,AttributeError,TypeError):pass
        if isinstance(code,str) and re.fullmatch(r'[A-Za-z0-9_\-]{1,80}',code) and key not in code:status['provider_error_code']=code
        normalized=code.lower() if isinstance(code,str) else ''
        status['failure_class']=('auth' if exc.code in (401,403) or normalized in ('invalidapikey','invalid_api_key','unauthorized','accessdenied') else 'billing' if exc.code==402 or normalized in ('insufficient_quota','insufficient_funds','payment_required','billing_hard_limit_reached','arrearage','insufficientbalance','insufficient_balance') else 'rate_limit' if exc.code==429 else 'transient')
    except CaptureStorageFailure:
        status.update(fatal_capture=True,error_type='CaptureStorageFailure')
    except NetworkFailure:
        status['error_type']='NetworkFailure'
    except Exception as exc:
        status.update(error_type=type(exc).__name__,fatal_capture=True)
    finally:
        _save(folder / 'worker-status.json', status)


def _stop_child(child):
    """Bounded cleanup on normal completion and every parent exception path."""
    if child.is_alive():
        try:
            child.terminate()
            child.join(1)
        finally:
            if child.is_alive():
                child.kill()
                child.join(1)
    if child.is_alive():
        raise RuntimeError('ChildShutdownFailed')


def capture_sse(folder, url, body, key, model, timeout_seconds):
    """Capture in one spawn child; parse only after confirmed child exit.

    Callers using spawn must invoke this from an importable module with the
    usual __main__ guard. Results describe transport only, never HTTP billing.
    Accepted DONE framing: LF/CRLF lines, an empty-line event terminator,
    exactly data:[DONE] or data: [DONE], optionally accompanied by comments.
    Other SSE field/whitespace forms conservatively remain incomplete.
    Authorization headers and exception diagnostics are never persisted;
    arbitrary successful response bytes (including remote credential echoes) are kept.
    HTTP error bodies alone are capped at 1MiB and exact-key redacted.
    """
    if type(timeout_seconds) not in (int, float) or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError('finite positive timeout required')
    started = time.monotonic()
    deadline = started + timeout_seconds
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    (folder / 'response.sse').open('xb').close()
    child = multiprocessing.get_context('spawn').Process(target=_capture_worker, args=(str(folder), url, body, key))
    timed_out = False
    try:
        child.start()
        child.join(max(0, deadline - time.monotonic()))
        timed_out = child.is_alive() or time.monotonic() >= deadline
        _stop_child(child)
        _save(folder/'lifecycle.json',dict(child_pid=child.pid,child_exitcode=child.exitcode,child_stopped=not child.is_alive(),timed_out=timed_out,seconds=time.monotonic()-started))
        spec = importlib.util.spec_from_file_location('immutable_teacher_parser', PARSER_PATH)
        parser = importlib.util.module_from_spec(spec); spec.loader.exec_module(parser)
        result = {}
        raw_bytes=(folder / 'response.sse').read_bytes()
        try:
            parser.parse_sse(raw_bytes.splitlines(), model, state=result)
        except Exception as exc:
            result['error_type'] = type(exc).__name__
        status_path = folder / 'worker-status.json'
        status = None
        headers_path=folder/'http-headers.json'
        if headers_path.exists():status=json.loads(headers_path.read_text(encoding='utf-8'))
        if status_path.exists():
            status = dict(status or {},**json.loads(status_path.read_text(encoding='utf-8')))
        if status is not None:
            for name in ('http_status', 'error_type','provider_error_code','failure_class','http_body_truncated','fatal_capture'):
                if name in status:
                    result[name] = status[name]
        fatal=capture_failure(status,child.exitcode,timed_out)
        status=status or {}
        result['fatal_capture']=fatal
        if not status.get('done_event') or child.exitcode != 0 or status.get('error_type'):
            result.update(complete=False)
            result.setdefault('error_type', 'IncompleteCompletion')
        if timed_out:
            result.update(complete=False, error_type='TimeoutError')
        result['transport_complete']=bool(status.get('done_event') and child.exitcode==0 and not status.get('error_type') and not timed_out)
        result.update(timed_out=timed_out, child_pid=child.pid, child_exitcode=child.exitcode,
                      child_stopped=not child.is_alive(), seconds=time.monotonic() - started)
        _save(folder / 'result.json', result)
        return result
    finally:
        if child.pid is not None:
            _stop_child(child)
            child.close()
