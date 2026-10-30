"""Explicit paths and exclusive JSON output for the new evaluation only."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
N100 = ROOT.parent / 'finance_program_student_n100_20261024_r2'
TRAIN = ROOT.parent / 'finance_program_student_fresh162_20261022_r5'
ADAPTER_SHA = '6d78e440ab0c9749427e002d54fd94cca06dea77c31b63ae3628b48fdbec4e64'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def put(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def native():
    # Historical modules import their own common/contract by these names.
    # This entrypoint is intended for a fresh process, like the original runner.
    sys.path.insert(0, str(N100))
    import eval_runtime
    return eval_runtime


def verify():
    freeze = json.loads((ROOT / 'freeze.json').read_bytes())
    for row in freeze['files'] + freeze['dependencies']:
        base = ROOT if row in freeze['files'] else ROOT.parent
        p = base / row['path']
        if p.is_symlink() or p.stat().st_size != row['bytes'] or sha(p) != row['sha256']:
            raise ValueError('frozen source differs: ' + str(p))
    return freeze
