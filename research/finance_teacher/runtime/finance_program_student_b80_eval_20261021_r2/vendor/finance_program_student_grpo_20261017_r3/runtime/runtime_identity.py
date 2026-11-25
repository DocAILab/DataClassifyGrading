"""Exclusive core/test byte identity for this explicit protocol variant."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity():
    files = [p for name in ('program_loop', 'vendor', 'tests') for p in (ROOT / name).rglob('*')
             if p.is_file() and '__pycache__' not in p.parts]
    files.append(ROOT / 'runtime_identity.py')
    entries = [{'path': p.relative_to(ROOT).as_posix(), 'sha256': sha(p)} for p in sorted(files)]
    signature = hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()
    return {'schema': 'grounded-runtime-v1', 'core_signature_sha256': signature, 'files': entries,
            'old_nontraining_comparable_without_rebaseline': False}


def verify():
    current = identity()
    assert current == json.loads((ROOT / 'reviews/runtime-manifest.json').read_text(encoding='utf-8'))
    return current


if __name__ == '__main__':
    path = ROOT / 'reviews/runtime-manifest.json'; path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:json.dump(identity(), stream, indent=2)
    print(json.dumps({'runtime_signature': verify()['core_signature_sha256'], 'files': len(verify()['files'])}))
