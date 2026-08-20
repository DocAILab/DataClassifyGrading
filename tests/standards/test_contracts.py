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


def _category(entry_id: str, level: str | None = None, row: int = 1, **kw) -> StandardCategory:
    return StandardCategory(
        standard_entry_id=entry_id,
        category_id=kw.get("category_id", entry_id),
        name=kw.get("name", entry_id),
        path=kw.get("path", (entry_id,)),
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
        entries=(
            _category("finance:业务.交易信息.交易通用信息.交易基本信息", "L2"),
            _category("finance:业务.账户信息..基本信息", "L1"),
        ),
    )
    mapping = standard.to_mapping()
    mapping["entries"] = list(reversed(mapping["entries"]))  # scramble order
    rebuilt = CanonicalStandard.from_mapping(mapping)
    assert rebuilt.dataset == standard.dataset
    assert rebuilt.id_strategy == standard.id_strategy
    assert rebuilt.fingerprint() == standard.fingerprint()


def test_round_trip_preserves_level_and_source():
    entry = StandardCategory(
        standard_entry_id="A1-1-1",
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
    rebuilt = StandardCategory.from_mapping(copy.deepcopy(entry.to_mapping()))
    assert rebuilt == entry


def test_fingerprint_independent_of_source_rows():
    a = _category("A", "L1", row=1)
    b = _category("A", "L1", row=99)  # same content, different source row
    sa = CanonicalStandard("s", "code", standard_source=SourceRef(), entries=(a,))
    sb = CanonicalStandard("s", "code", standard_source=SourceRef(), entries=(b,))
    assert sa.fingerprint() == sb.fingerprint()


def test_training_projection_groups_entries_by_category_id():
    standard = CanonicalStandard(
        dataset="finance",
        id_strategy="path",
        standard_source=SourceRef(),
        entries=(
            _category("finance:业务.合约协议.合同通用信息.基本信息", "L2", category_id="finance:业务.合约协议.基本信息"),
            _category("finance:业务.合约协议.贷款业务信息.基本信息", "L2", category_id="finance:业务.合约协议.基本信息"),
            _category("finance:业务.交易信息.交易通用信息.交易基本信息", "L2", category_id="finance:业务.交易信息.交易基本信息"),
        ),
    )
    projection = standard.training_projection()
    assert standard.trainable_category_count() == 2
    assert projection["finance:业务.合约协议.基本信息"] == [
        "finance:业务.合约协议.合同通用信息.基本信息",
        "finance:业务.合约协议.贷款业务信息.基本信息",
    ]
    assert len(standard.entries_by_category_id()["finance:业务.合约协议.基本信息"]) == 2
