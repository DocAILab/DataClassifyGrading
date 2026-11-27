"""Exact R13 transport, only this execution partition HTTP lease is new."""
import types
from pathlib import Path
from r14_common import ROOT, R13, load

source = load('r14_pinned_R13_wire', R13/'r13_wire.py')
isolation = load('r14_pinned_OS_isolation', R13/'r13_isolation.py')
HTTP_LEASE = ROOT/'active-http.lock'
child_args = source.child_args


def _worker(folder, url, body_path, key):
    source.old._save(Path(folder)/'wire-child-before-lease.json',
                     dict(stage='imported_before_HTTP_lease', new_HTTP=0))
    with isolation.lease(HTTP_LEASE):
        source.old._save(Path(folder)/'wire-child-after-lease.json',
                         dict(stage='HTTP_lease_acquired_before_transport', new_HTTP=0))
        source.old._capture_worker(folder, url, Path(body_path).read_bytes(), key)


namespace = dict(source.capture_sse.__globals__)
namespace['_worker'] = _worker
capture_sse = types.FunctionType(source.capture_sse.__code__, namespace, 'capture_sse',
                                source.capture_sse.__defaults__)
