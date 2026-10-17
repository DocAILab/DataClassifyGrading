"""Behavior tests for the independent source-snapshot audit."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


def audit_function():
    path = Path(__file__).resolve().parents[1] / 'tools' / 'verify_source.py'
    assert path.is_file(), 'source-integrity implementation is not present yet'
    spec = importlib.util.spec_from_file_location('source_integrity_tool', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.verify_manifest


def fixture_manifest(tmp_path):
    vendor = tmp_path / 'vendor'
    vendor.mkdir()
    original = tmp_path / 'original'
    original.mkdir()
    payload = b'print("original")\n'
    (vendor / 'core.py').write_bytes(payload)
    (original / 'core.py').write_bytes(payload)
    entry = {'path': 'core.py', 'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}
    manifest = tmp_path / 'source.json'
    manifest.write_text(json.dumps({'schema': 1, 'source': str(original), 'files': 1, 'entries': [entry]}), encoding='utf-8')
    return manifest, vendor, original


def test_exact_snapshot_and_original_pass(tmp_path):
    manifest, vendor, _ = fixture_manifest(tmp_path)
    result = audit_function()(manifest, vendor, verify_original=True)
    assert result['verified'] is True
    assert result['files'] == 1
    assert result['original_verified'] is True


def test_tampered_snapshot_is_rejected(tmp_path):
    manifest, vendor, _ = fixture_manifest(tmp_path)
    (vendor / 'core.py').write_bytes(b'changed')
    with pytest.raises(ValueError, match='core.py'):
        audit_function()(manifest, vendor)


def test_extra_source_file_is_rejected(tmp_path):
    manifest, vendor, _ = fixture_manifest(tmp_path)
    (vendor / 'extra.py').write_bytes(b'new')
    with pytest.raises(ValueError, match='inventory'):
        audit_function()(manifest, vendor)


def test_original_tampering_is_independently_rejected(tmp_path):
    manifest, vendor, original = fixture_manifest(tmp_path)
    (original / 'core.py').write_bytes(b'changed')
    with pytest.raises(ValueError, match='original'):
        audit_function()(manifest, vendor, verify_original=True)


def test_manifest_parent_traversal_is_rejected(tmp_path):
    manifest, vendor, _ = fixture_manifest(tmp_path)
    data = json.loads(manifest.read_text(encoding='utf-8'))
    data['entries'][0]['path'] = '../core.py'
    manifest.write_text(json.dumps(data), encoding='utf-8')
    with pytest.raises(ValueError, match='path'):
        audit_function()(manifest, vendor)
