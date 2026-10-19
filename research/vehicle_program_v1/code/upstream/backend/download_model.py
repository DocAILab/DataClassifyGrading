"""Pinned, token-free official checkpoint download; never executes remote code."""
import hashlib
import json
from pathlib import Path
import re

REPO = 'Qwen/Qwen3.5-9B'
MODEL_DIR = Path(__file__).resolve().parent / 'models' / 'Qwen3.5-9B'
ALLOWED = {'config.json', 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json',
           'special_tokens_map.json', 'chat_template.json', 'chat_template.jinja', 'vocab.json',
           'merges.txt', 'added_tokens.json', 'model.safetensors.index.json'}


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def allowed(name):
    return isinstance(name, str) and (name in ALLOWED or re.fullmatch(r'model(?:\.safetensors)?(?:-\d+-of-\d+)?\.safetensors', name) is not None)


def validate_identity(identity):
    if identity.get('repo_id') != REPO or not re.fullmatch('[0-9a-f]{40}', identity.get('revision', '')):
        raise ValueError('invalid official model identity')
    entries = identity.get('files', [])
    names = [e.get('path') for e in entries]
    if not entries or len(set(names)) != len(names) or not all(allowed(n) for n in names):
        raise ValueError('invalid model identity file inventory')
    if 'config.json' not in names or 'tokenizer.json' not in names or not any(n.endswith('.safetensors') for n in names):
        raise ValueError('incomplete model identity')
    if any(type(e.get('bytes')) is not int or e['bytes'] < 0 for e in entries):
        raise ValueError('model identity requires expected file sizes')
    if any(e.get('expected_sha256') is not None and not re.fullmatch('[0-9a-f]{64}', e['expected_sha256']) for e in entries):
        raise ValueError('invalid model identity expected hash')


def download_model(model_dir=MODEL_DIR, *, api=None, snapshot=None):
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = model_dir / 'model-manifest.json'
    if manifest_path.exists():
        raise FileExistsError(f'completed manifest already exists: {manifest_path}')
    identity_path = model_dir / 'download-identity.json'
    if api is None or snapshot is None:
        from huggingface_hub import HfApi, snapshot_download
        api = api or HfApi(token=False)
        snapshot = snapshot or snapshot_download
    if identity_path.exists():
        identity = json.loads(identity_path.read_text(encoding='utf-8'))
    else:
        if any(model_dir.iterdir()):
            raise ValueError('unidentified existing checkpoint directory')
        info = api.model_info(REPO, files_metadata=True, token=False)
        identity = {'repo_id': REPO, 'revision': info.sha,
                    'files': [{'path': f.rfilename, 'bytes': f.size,
                               'expected_sha256': getattr(getattr(f, 'lfs', None), 'sha256', None)}
                              for f in info.siblings if allowed(f.rfilename)]}
        validate_identity(identity)
        write_json(identity_path, identity)
    validate_identity(identity)
    snapshot(repo_id=REPO, revision=identity['revision'], local_dir=str(model_dir),
             allow_patterns=[e['path'] for e in identity['files']], token=False)
    files = []
    for expected in identity['files']:
        path = model_dir / expected['path']
        if path.is_symlink() or not path.is_file() or path.stat().st_size != expected['bytes']:
            raise ValueError(f'download file size/path mismatch: {expected["path"]}')
        actual_hash = sha256(path)
        if expected.get('expected_sha256') and actual_hash != expected['expected_sha256']:
            raise ValueError(f'download file trusted hash mismatch: {expected["path"]}')
        files.append({**expected, 'sha256': actual_hash})
    manifest = {'repo_id': REPO, 'revision': identity['revision'], 'total_bytes': sum(e['bytes'] for e in files),
                'download_identity_sha256': sha256(identity_path), 'files': files}
    expected = {e['path'] for e in files} | {'download-identity.json'}
    actual = {p.relative_to(model_dir).as_posix() for p in model_dir.rglob('*') if p.is_file() and '.cache' not in p.relative_to(model_dir).parts}
    if actual != expected:
        raise ValueError('download file inventory mismatch')
    write_json(manifest_path, manifest)
    verify_model(model_dir)
    return manifest


def verify_model(model_dir):
    """Read-only complete inventory, byte and hash gate before model loading."""
    model_dir = Path(model_dir)
    identity_path = model_dir / 'download-identity.json'
    identity = json.loads(identity_path.read_text(encoding='utf-8'))
    manifest = json.loads((model_dir / 'model-manifest.json').read_text(encoding='utf-8'))
    validate_identity(identity)
    validate_identity(manifest)
    if manifest['revision'] != identity['revision'] or manifest['download_identity_sha256'] != sha256(identity_path):
        raise ValueError('model identity hash/revision mismatch')
    if [{k: v for k, v in e.items() if k != 'sha256'} for e in manifest['files']] != identity['files']:
        raise ValueError('model identity inventory mismatch')
    expected = {e['path'] for e in manifest['files']} | {'download-identity.json', 'model-manifest.json'}
    actual = {p.relative_to(model_dir).as_posix() for p in model_dir.rglob('*') if p.is_file() and '.cache' not in p.relative_to(model_dir).parts}
    if actual != expected:
        raise ValueError('model file inventory mismatch')
    for entry in manifest['files']:
        path = model_dir / entry['path']
        if path.is_symlink() or path.stat().st_size != entry['bytes'] or sha256(path) != entry['sha256'] or (entry.get('expected_sha256') and entry['expected_sha256'] != entry['sha256']):
            raise ValueError(f'model file hash/size mismatch: {entry["path"]}')
    return manifest


if __name__ == '__main__':
    result = download_model()
    print(json.dumps({'repo_id': result['repo_id'], 'revision': result['revision'], 'total_bytes': result['total_bytes']}))
