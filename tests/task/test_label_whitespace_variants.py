"""Read-only check for suspicious whitespace-only label variants.

This check is intentionally READ-ONLY: it detects classification labels where
removing all whitespace produces the same key but the raw strings differ
(e.g. ``"经营 管理"`` vs ``"经营管理"``) and REPORTS them. It never
auto-repairs labels — label fixes are manual upstream edits in
``data/<dataset>/all.json`` (the formal input), and this test exists so a
re-introduced variant is caught before downstream artifacts are regenerated.

Two tiers, matching the repo's CI convention:

- logic tier: runs always on inline data (CI-safe),
- data tier: skips when ``data/<dataset>/all.json`` is unavailable locally.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
CLASSIFICATION_FIELDS = ("level_1", "level_2", "level_3", "level_4")
DATASETS = ("finance", "infra", "pers_info", "shougang")


def find_whitespace_variants(
    labels: list[str],
) -> dict[str, list[str]]:
    """Return {compact_key: sorted distinct raw strings} for labels whose
    whitespace-stripped key collapses multiple distinct raw forms.

    ``compact_key`` = all whitespace removed. Only groups with more than one
    distinct raw string are reported.
    """
    grouped: dict[str, set[str]] = defaultdict(set)
    for label in labels:
        if not label:  # missing/empty level must not be reported as a variant
            continue
        grouped[re.sub(r"\s+", "", label)].add(label)
    return {
        key: sorted(raws)
        for key, raws in grouped.items()
        if len(raws) > 1
    }


def _all_classification_labels(records: list[dict], fields) -> list[str]:
    labels: list[str] = []
    for record in records:
        classification = record.get("classification") or {}
        for field in fields:
            value = classification.get(field)
            if value:
                labels.append(str(value))
    return labels


# ---------------------------------------------------------------------------
# logic tier (CI-safe, inline data)
# ---------------------------------------------------------------------------


def test_find_whitespace_variants_detects_inner_space() -> None:
    variants = find_whitespace_variants(
        ["经营 管理", "经营管理", "业务", "客户", "经营管理"]
    )
    assert variants == {
        "经营管理": ["经营 管理", "经营管理"],
    }


def test_find_whitespace_variants_multiple_whitespace_kinds() -> None:
    # tabs, full-width spaces, and multiple runs all collapse to one key.
    variants = find_whitespace_variants(
        ["经营\t管理", "经营管理", "经营  管理", "客户"]
    )
    assert variants == {
        "经营管理": ["经营\t管理", "经营  管理", "经营管理"],  # \t sorts before space
    }


def test_find_whitespace_variants_clean_labels_report_nothing() -> None:
    assert find_whitespace_variants(["业务", "经营管理", "客户"]) == {}
    assert find_whitespace_variants([]) == {}
    assert find_whitespace_variants(["", "", None or "客户"]) == {}


# ---------------------------------------------------------------------------
# data tier (read-only over real all.json; skips when data unavailable)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dataset", DATASETS)
def test_real_dataset_has_no_whitespace_variant_labels(dataset: str) -> None:
    record_path = DATA_DIR / dataset / "all.json"
    if not record_path.is_file():
        pytest.skip(f"data/{dataset}/all.json not available")

    with record_path.open(encoding="utf-8") as handle:
        records = json.load(handle)
    assert isinstance(records, list) and records, f"{dataset}/all.json not a non-empty list"

    labels = _all_classification_labels(records, CLASSIFICATION_FIELDS)
    variants = find_whitespace_variants(labels)
    assert variants == {}, (
        f"{dataset}: whitespace-only label variants present (data quality "
        "issue — fix the label in the upstream input, do not auto-repair): "
        f"{variants}"
    )
