"""Phase 1 canonical standard — raw-source checksum verification (hermetic).

Covers Blocker-2 option B: CLI refuses to build from a missing manifest /
mismatched file, so a silently wrong raw workbook can never produce a
"canonical" standard. Uses tmp files only (no repo data dependency).
"""

from __future__ import annotations

import hashlib
import json

import pytest

from script.standard.cli import _verify_checksums


def _write(path, text: bytes):
    path.write_bytes(text)
    return hashlib.sha256(text).hexdigest()


def test_checksum_mismatch_raises(tmp_path):
    manifest = tmp_path / "checksums.json"
    good = tmp_path / "a.xlsx"
    expected = _write(good, b"A" * 100)

    other = tmp_path / "b.xlsx"
    _write(other, b"B" * 100)
    manifest.write_text(json.dumps({"a.xlsx": expected}), encoding="utf-8")

    with pytest.raises(ValueError, match="sha256 mismatch"):
        _verify_checksums({"a.xlsx": other}, manifest, allow_skip=False)


def test_checksum_match_passes(tmp_path):
    manifest = tmp_path / "checksums.json"
    good = tmp_path / "a.xlsx"
    expected = _write(good, b"A" * 100)
    manifest.write_text(json.dumps({"a.xlsx": expected}), encoding="utf-8")
    _verify_checksums({"a.xlsx": good}, manifest, allow_skip=False)  # no raise


def test_checksum_missing_manifest_requires_override(tmp_path):
    manifest = tmp_path / "checksums.json"  # deliberately absent
    good = tmp_path / "a.xlsx"
    _write(good, b"A" * 100)
    with pytest.raises(FileNotFoundError, match="checksum manifest not found"):
        _verify_checksums({"a.xlsx": good}, manifest, allow_skip=False)
    # explicit override is allowed for offline tweaks
    _verify_checksums({"a.xlsx": good}, manifest, allow_skip=True)
