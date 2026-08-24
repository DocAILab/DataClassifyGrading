"""Deterministic checkpoint fingerprinting for reference-checkpoint lineage."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from script.verl.sft.record_checkpoint import (
    build_provenance,
    main,
    tree_hash,
)


def _make_checkpoint(root: Path) -> Path:
    (root / "model").mkdir(parents=True)
    (root / "model" / "weights.safetensors").write_bytes(b"\x00\x01\x02")
    (root / "config.json").write_text("{\"demo\": true}", encoding="utf-8")
    return root


def test_tree_hash_is_content_and_layout_sensitive(tmp_path) -> None:
    first = _make_checkpoint(tmp_path / "a")
    second = _make_checkpoint(tmp_path / "b")
    assert tree_hash(first)[0] == tree_hash(second)[0]

    # any content change changes the digest
    (second / "config.json").write_text("{\"demo\": false}", encoding="utf-8")
    assert tree_hash(first)[0] != tree_hash(second)[0]

    # renaming a file also changes the digest (layout participates)
    renamed = _make_checkpoint(tmp_path / "c")
    (renamed / "config.json").rename(renamed / "renamed.json")
    assert tree_hash(_make_checkpoint(tmp_path / "d"))[0] != tree_hash(renamed)[0]


def test_provenance_links_export_report(tmp_path) -> None:
    checkpoint = _make_checkpoint(tmp_path / "ckpt")
    export_report = tmp_path / "export_report.json"
    payload = {"label_gap_gate": {"status": "passed"}}
    export_report.write_text(json.dumps(payload), encoding="utf-8")

    provenance = build_provenance(checkpoint, export_report)
    assert provenance["algorithm"] == "sha256-tree-v1"
    assert provenance["files"] == 2
    assert provenance["total_bytes"] > 0
    assert provenance["training_export_report"]["sha256"] == (
        hashlib_sha256(export_report)
    )
    # deterministic: identical inputs give identical fingerprint
    again = build_provenance(checkpoint, export_report)
    assert again["checkpoint_sha256"] == provenance["checkpoint_sha256"]


def hashlib_sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_cli_writes_provenance_atomically(tmp_path) -> None:
    checkpoint = _make_checkpoint(tmp_path / "ckpt")
    output = tmp_path / "provenance.json"
    common = [
        "--checkpoint-dir", str(checkpoint),
        "--output", str(output),
    ]
    assert main(common) == 0
    stored = json.loads(output.read_text(encoding="utf-8"))
    assert stored["checkpoint_sha256"] == build_provenance(checkpoint)["checkpoint_sha256"]

    assert main(common) == 2  # refuses overwrite without flag
    first = output.read_text(encoding="utf-8")
    assert main([*common, "--overwrite"]) == 0
    assert output.read_text(encoding="utf-8") == first  # deterministic content


def test_missing_checkpoint_dir_fails(tmp_path) -> None:
    with pytest.raises(NotADirectoryError):
        build_provenance(tmp_path / "nope")


def test_empty_checkpoint_dir_rejected(tmp_path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no files"):
        build_provenance(empty)
