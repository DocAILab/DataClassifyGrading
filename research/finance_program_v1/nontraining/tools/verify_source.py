"""Independent SHA/inventory audit of the reused, immutable executor snapshot."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath


SOURCE_EXTENSIONS = {'.py', '.md', '.json'}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_manifest(manifest_path, vendor_root, *, verify_original=False):
    manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    root = Path(vendor_root).resolve()
    expected = {}
    for entry in manifest['entries']:
        relative = entry['path']
        pure = PurePosixPath(relative)
        if (not isinstance(relative, str) or not relative or '\\' in relative
                or pure.is_absolute() or '..' in pure.parts or ':' in relative
                or pure.as_posix() != relative or relative in expected):
            raise ValueError(f'invalid manifest path: {relative!r}')
        expected[relative] = entry
    if manifest['files'] != len(expected):
        raise ValueError('manifest inventory count mismatch')
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*')
              if p.is_file() and p.suffix in SOURCE_EXTENSIONS}
    if actual != set(expected):
        raise ValueError(f'source inventory mismatch: missing={sorted(set(expected)-actual)}, extra={sorted(actual-set(expected))}')
    for relative, entry in expected.items():
        path = root / relative
        if not path.resolve().is_relative_to(root):
            raise ValueError(f'source path escapes vendor: {relative}')
        if path.stat().st_size != entry['bytes'] or sha256(path) != entry['sha256']:
            raise ValueError(f'source SHA/size mismatch: {relative}')
        if verify_original:
            original = Path(manifest['source']) / relative
            if not original.is_file() or sha256(original) != entry['sha256']:
                raise ValueError(f'original source mismatch: {relative}')
    return {'verified': True, 'files': len(expected), 'original_verified': verify_original}


def main():
    task_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=task_root / 'reviews/source-manifest.json')
    parser.add_argument('--vendor', type=Path, default=task_root / 'vendor')
    parser.add_argument('--verify-original', action='store_true')
    args = parser.parse_args()
    print(json.dumps(verify_manifest(args.manifest, args.vendor, verify_original=args.verify_original)))


if __name__ == '__main__':
    main()
