"""Alignment CLI tests: legacy corpus discovery keys after the data-layout
migration (data/legacy/*.corpus.json).

Regression guard for the ``Path.stem`` double-suffix bug: a file named
``finance.corpus.json`` must register as ``corpus:finance`` (the key that
``build_schema_issues`` hardcodes), never ``corpus:finance.corpus``.
"""

from __future__ import annotations

import json

import pytest

from script.analysis.analyze_dataset_corpus_alignment import (
    DEFAULT_LEGACY_DIR,
    discover_corpora,
)


@pytest.fixture
def legacy_dir(tmp_path, monkeypatch):
    (tmp_path / "finance.corpus.json").write_text(
        json.dumps({"category1": "definition one"}), encoding="utf-8"
    )
    (tmp_path / "shougang.corpus.json").write_text(
        json.dumps({"category2": "definition two"}), encoding="utf-8"
    )
    monkeypatch.setattr(
        "script.analysis.analyze_dataset_corpus_alignment.DEFAULT_LEGACY_DIR",
        tmp_path,
    )
    return tmp_path


def test_legacy_corpus_keys_use_dataset_name_not_double_suffix(legacy_dir):
    corpora = discover_corpora()
    assert "corpus:finance" in corpora
    assert "corpus:shougang" in corpora
    # the Path.stem double-suffix ("finance.corpus") must never be registered
    assert "corpus:finance.corpus" not in corpora
    assert "corpus:shougang.corpus" not in corpora
    for key in corpora:
        assert not key.endswith(".corpus"), f"bad legacy key {key!r}"


def test_legacy_corpus_stem_matches_hardcoded_consumer_key(legacy_dir):
    corpora = discover_corpora()
    # build_schema_issues() looks up standard_stats["corpus:finance"]
    assert "corpus:finance" in corpora
