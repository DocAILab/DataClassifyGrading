"""Exclusive evidence writes and frozen-file identity checks; no credentials."""
import hashlib
import json
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def put(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def verify_manifest(root):
    root = Path(root)
    m = read(root / 'manifest.json')
    for item in m['files']:
        p = root / item['path']
        if not p.is_relative_to(root) or digest(p) != item['sha256']:
            raise ValueError('frozen file SHA mismatch: ' + item['path'])
    if digest(root / 'data/planner200.jsonl') != m['config']['data_sha256']:
        raise ValueError('data identity mismatch')
    return m
