"""Additive launch-only namespace fix; frozen inference/runtime files unchanged."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent

def resolve_supervisor():
    spec = importlib.util.spec_from_file_location('frozen_r8_launch_wrapper', ROOT / 'launch_smoke.py')
    wrapper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wrapper)
    native = wrapper.namespace['namespace']
    native.update(verify=wrapper.verify, put=wrapper.put)
    if not callable(native.get('main')) or native['main'].__globals__ is not native:
        raise ValueError('actual native supervisor namespace mismatch')
    return native

def verify_entry():
    lock = json.loads((ROOT / 'reviews/entry-lock-01.json').read_bytes())
    for entry in lock['files']:
        path = ROOT / entry['path']
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError('additive entry changed')

def watch():
    verify_entry()
    native = resolve_supervisor()
    native['verify']()
    folder = ROOT / 'launch'
    folder.mkdir(exist_ok=False)
    with (folder/'stdout.log').open('xb') as out, (folder/'stderr.log').open('xb') as err:
        worker = subprocess.Popen([native['PYTHON'], '-B', str(ROOT/'supervisor_entry.py'), '--main'],
            cwd=ROOT, stdin=subprocess.DEVNULL, stdout=out, stderr=err)
        native['put'](folder/'started.json', dict(watcher_pid=__import__('os').getpid(),
                         supervisor_pid=worker.pid, time_unix=time.time()))
        code = worker.wait()
    native['put'](folder/'exit.json', dict(actual_returncode=code, time_unix=time.time()))
    return code

if __name__ == '__main__':
    if sys.argv[1:] == ['--watch']:
        raise SystemExit(watch())
    elif sys.argv[1:] == ['--main']:
        verify_entry()
        raise SystemExit(resolve_supervisor()['main']())
    else:
        raise SystemExit('explicit --watch or --main required')
