"""Exclusive artifact writes; byte identity and runtime import bootstrap."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / 'runtime'), str(ROOT / 'legacy')]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):h.update(b)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def put(path, value):
    path = Path(path);path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False);f.write('\n')


def verify(root=ROOT):
    root = Path(root)
    m = read(root / 'manifest.json')
    for item in m['files']:
        p = root / item['path']
        if not p.resolve().is_relative_to(root.resolve()) or sha(p) != item['sha256']:
            raise ValueError('evaluation frozen file SHA differs: ' + item['path'])
    return m
