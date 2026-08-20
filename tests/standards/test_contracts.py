"""Phase 1 canonical standard — contract unit tests (hermetic, no IO / Excel)."""

from __future__ import annotations

import copy

import pytest

from agent.standards.contracts import (
    CanonicalStandard,
    SourceRef,
    StandardCategory,
    compact,
    normalize_standard_level,
    strip_code,
)


def _category(category_id: str, name: str, level: str | None = None, row: int = 1) -> StandardCategory:
    return StandardCategory(
        category_id=category_id,
        name=name,
        path=(name,),
        description="",
        code=None,
        standard_data_level=level,
        raw_level=level or "",
        source=SourceRef(file="f.xlsx", sheet="s", row=row),
    )


def test_normalize_standard_level_valid_forms():
    for raw, expected in {
        "1": "L1", "2": "L2", "3": "L3", "4": "L4",
        "L1": "L1", "LEVEL3": "L3", "1级": "L1", "4级": "L4",
    }.items():
        level, kept = normalize_standard_level(raw)
        assert level == expected
        assert kept == raw.strip()
    assert normalize_standard_level(None) == (None, "")
    assert normalize_standard_level("") == (None, "")


def test_normalize_standard_level_unparseable_never_guessed():
    # finance guide anomalies: 'l' (typo of 1) and '3 4' (ambiguous) must stay
    # None and keep the raw text — the builder reports them, never fixes.
    for raw in ("l", "3 4", "敏感级", "高"):
        level, kept = normalize_standard_level(raw)
        assert level is None
        assert kept == raw


def test_compact_and_strip_code():
    assert compact("经营 管理　技术") == "经营管理技术"
    assert strip_code("科研设备预约管理（A1-1-1）") == "科研设备预约管理"
    assert strip_code("生产数据域\n（B）") == "生产数据域"
    assert strip_code("基本信息（公开）") == "基本信息（公开）"  # parens without code kept


def test_standard_round_trip_stable():
    standard = CanonicalStandard(
        dataset="finance",
        id_strategy="path",
        standard_name="指南",
        standard_source=SourceRef(file="data/raw/f.xlsx", sheet="Table 1", row=None),
        categories=(_category("finance:a.b.c", "c", "L3"), _category("finance:x", "x", "L1")),
    )
    mapping = standard.to_mapping()
    mapping["categories"] = sorted(
        mapping["categories"], key=lambda c: c["category_id"], reverse=True
    )  # scramble order
    rebuilt = CanonicalStandard.from_mapping(mapping)
    assert rebuilt.dataset == standard.dataset
    assert rebuilt.id_strategy == standard.id_strategy
    assert rebuilt.fingerprint() == standard.fingerprint()


def test_round_trip_preserves_level_and_source():
    category = StandardCategory(
        category_id="A1-1-1",
        name="科研设备预约管理",
        path=("研发数据域", "产品研发", "科研检验", "科研设备预约管理"),
        description="描述",
        code="A1-1-1",
        standard_data_level="L3",
        raw_level="3",
        content="资源说明",
        source=SourceRef(file="x.xlsx", sheet="数据分类分级", row=7),
    )
    rebuilt = StandardCategory.from_mapping(copy.deepcopy(category.to_mapping()))
    assert rebuilt == category


def test_fingerprint_independent_of_source_rows():
    a = _category("A", "a", "L1", row=1)
    b = _category("A", "a", "L1", row=99)  # same content, different source row
    sa = CanonicalStandard("s", "code", standard_source=SourceRef(), categories=(a,))
    sb = CanonicalStandard("s", "code", standard_source=SourceRef(), categories=(b,))
    assert sa.fingerprint() == sb.fingerprint()
