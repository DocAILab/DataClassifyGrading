"""Pure SHA inventories. No model acceptance, transport, launch, or approval.

Create: python -B -m tools.freeze_revision --root . --output reviews/revision-identity
Verify: python -B -m tools.freeze_revision --root . --output reviews/revision-identity --verify
The output directory is created exclusively; a failed/old freeze is never reused.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IDENTITY = ROOT / 'reviews/revision-identity'
SOURCE_DIRS = ('program_loop', 'vendor', 'tools', 'tests', 'tests_batch')
CORE_DIRS = ('program_loop', 'vendor', 'tools')
SOURCE_FILES = ('reviews/independent_audit_01.py', 'reviews/remote_gate.py')
EXECUTION_FILES = ('data/study.json', 'data/study-manifest.json', 'reviews/trusted-model-manifest-01.json')
BASELINE_FILES = {'source_manifest_sha256': 'reviews/baseline-source-manifest-01.json',
                  'execution_manifest_sha256': 'reviews/baseline-execution-manifest-01.json'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def signature(entries):
    return hashlib.sha256(''.join(f"{e['path']}\t{e['bytes']}\t{e['sha256']}\n"
        for e in sorted(entries, key=lambda e: e['path'])).encode()).hexdigest()


def safe_path(root, name):
    if (not isinstance(name, str) or not name or '\\' in name or ':' in name or
        PurePosixPath(name).is_absolute() or PurePosixPath(name).as_posix() != name or
        any(p in ('', '.', '..') or p.endswith((' ', '.')) for p in name.split('/')) or
        any(ord(c) < 32 for c in name)):
        raise ValueError(f'unsafe inventory path: {name!r}')
    path = root
    for part in name.split('/'):
        path = path / part
        if path.is_symlink() or (hasattr(path, 'is_junction') and path.is_junction()):
            raise ValueError(f'symlink inventory path: {name}')
    # Canonical relative components and the ancestor link checks above prevent
    # escaping root without re-resolving the full Windows path for every file.
    return path


def inventory(root, names):
    entries = []
    seen = set()
    for name in sorted(names):
        path = safe_path(root, name)
        if name.casefold() in seen:
            raise ValueError('duplicate inventory path')
        seen.add(name.casefold())
        if not path.is_file():
            raise ValueError(f'missing inventory file: {name}')
        entries.append({'path': name, 'bytes': path.stat().st_size, 'sha256': sha(path)})
    return entries


def source_names(root):
    names = [p.name for p in root.glob('*.py')]
    def walk(directory):
        for path in directory.iterdir():
            relative = path.relative_to(root).as_posix()
            safe_path(root, relative)
            if path.name == '__pycache__':
                continue
            if path.is_dir():
                walk(path)
            elif path.is_file():
                names.append(relative)
            else:
                raise ValueError(f'nonregular inventory entry: {relative}')
    for name in SOURCE_DIRS:
        directory = safe_path(root, name)
        if not directory.is_dir():
            raise ValueError(f'missing source directory: {name}')
        walk(directory)
    return names + list(SOURCE_FILES)


def manifests(root):
    root = Path(root)
    if root.is_symlink() or (hasattr(root, 'is_junction') and root.is_junction()):
        raise ValueError('symlink root')
    provenance = {key: sha(safe_path(root, name)) for key, name in BASELINE_FILES.items()}
    source = inventory(root, source_names(root))
    execution = sorted(source + inventory(root, EXECUTION_FILES), key=lambda e: e['path'])
    shared = {'schema': 1, 'baseline_provenance': provenance}
    return ({**shared, 'kind': 'source', 'files': source, 'signature_sha256': signature(source)},
            {**shared, 'kind': 'execution', 'files': execution, 'signature_sha256': signature(execution),
             'source_signature_sha256': signature(source), 'study_sha256': sha(safe_path(root, 'data/study.json'))})


def environment_identity(root=ROOT):
    source, execution = manifests(root)
    return identity_from_manifests(source, execution)


def identity_from_manifests(source, execution):
    return {'revision_identity': {'schema': 1, 'source_signature_sha256': source['signature_sha256'],
            'execution_signature_sha256': execution['signature_sha256'],
            'study_sha256': execution['study_sha256'], 'baseline_provenance': source['baseline_provenance']},
            'core_source_sha256': {e['path']: e['sha256'] for e in source['files'] if e['path'].split('/')[0] in CORE_DIRS}}


def check_entries(root, manifest):
    entries = manifest.get('files')
    if not isinstance(entries, list) or not entries:
        raise ValueError('empty or invalid manifest inventory')
    for entry in entries:
        if (not isinstance(entry, dict) or set(entry) != {'path', 'bytes', 'sha256'} or
            type(entry['bytes']) is not int or entry['bytes'] < 0 or
            not isinstance(entry['sha256'], str) or not re.fullmatch('[0-9a-f]{64}', entry['sha256'])):
            raise ValueError('invalid inventory entry')
    actual = inventory(root, [e['path'] for e in entries])
    if actual != entries or signature(entries) != manifest.get('signature_sha256'):
        raise ValueError('inventory bytes/hash/order/signature mismatch')


def no_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate manifest key')
        result[key] = value
    return result


def verify_revision(root=ROOT, output=None):
    root = Path(root); output = Path(output) if output is not None else DEFAULT_IDENTITY
    expected = manifests(root)
    for name, wanted in zip(('source.json', 'execution.json'), expected):
        # Also reject linked manifest directories, not just linked final files.
        for parent in (output, *output.parents):
            if parent.is_symlink() or (hasattr(parent, 'is_junction') and parent.is_junction()):
                raise ValueError('symlink manifest directory')
        path = safe_path(output, name)
        actual = json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=no_duplicate_keys)
        check_entries(root, actual)
        if actual != wanted:
            raise ValueError(f'exact {name} inventory or provenance mismatch')
    return identity_from_manifests(*expected)


def freeze(root=ROOT, output=DEFAULT_IDENTITY):
    root = Path(root); output = Path(output)
    source, execution = manifests(root)
    for parent in (output, *output.parents):
        if parent.is_symlink() or (hasattr(parent, 'is_junction') and parent.is_junction()):
            raise ValueError('symlink output path')
    relative = output.resolve().relative_to(root.resolve()) if output.resolve().is_relative_to(root.resolve()) else None
    if relative and (relative.parts[0] in SOURCE_DIRS or relative.as_posix() in SOURCE_FILES + EXECUTION_FILES):
        raise ValueError('output would enter its own inventory')
    output.mkdir()  # Parent must exist; never merge or overwrite another revision.
    for name, value in (('source.json', source), ('execution.json', execution)):
        with (output / name).open('x', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
    return verify_revision(root, output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args(argv)
    result = verify_revision(args.root, args.output) if args.verify else freeze(args.root, args.output)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
