"""Phase 1 canonical standard — finance build tests (dict fixtures, hermetic).

Builders accept plain dict entries ({"level_1","level_2","level_3","leaf",
"description","content","raw_level","sheet","row"}) with no Excel dependency.
"""

from __future__ import annotations

from agent.standards.build import build_finance_standard


def _entry(**kw):
    base = dict(sheet="Table 1", row=1, content="", raw_level="")
    base.update(kw)
    return base


def _by_category(standard):
    """Find the single entry of a category (helper for fixtures)."""
    return {entry.category_id: entry for entry in standard.entries}


def test_finance_lossless_path_keeps_real_depth_and_no_padding():
    entries = [
        _entry(
            level_1="客户", level_2="个人", level_3="个人自然信息",
            leaf="个人基本概况信息", description="指个人基本情况数据",
            raw_level="3", row=3,
        ),
        _entry(
            level_1="业务", level_2="账户信息", level_3="",
            leaf="基本信息", description="账户基本信息", raw_level="2", row=40,
        ),
    ]
    standard, report = build_finance_standard(
        entries, source_file="data/raw/g.xlsx", source_sheet="Table 1"
    )
    by_category = _by_category(standard)
    full = by_category["finance:客户.个人.个人基本概况信息"]
    assert full.path == ("客户", "个人", "个人自然信息", "个人基本概况信息")  # 4 real levels
    assert full.standard_entry_id == "finance:客户.个人.个人自然信息.个人基本概况信息"
    assert full.standard_data_level == "L3"
    assert full.description == "指个人基本情况数据"
    shallow = by_category["finance:业务.账户信息.基本信息"]
    assert shallow.path == ("业务", "账户信息", "基本信息")  # empty 三级 omitted, no padding
    assert shallow.standard_entry_id == "finance:业务.账户信息..基本信息"  # empty slot kept
    assert shallow.standard_data_level == "L2"
    assert report.issues == []


def test_finance_identical_leaf_under_three_different_level3_kept_as_entries():
    # The Phase-1 contract keeps every real standard row LOSSESS: two rows with
    # the same L1/L2/leaf but different 三级子类 stay two entries that share a
    # training category_id (projection), instead of collapsing to one.
    entries = [
        _entry(
            level_1="客户", level_2="个人", level_3="个人自然信息",
            leaf="个人基本概况信息", raw_level="3", row=3,
        ),
        _entry(
            level_1="客户", level_2="个人", level_3="个人健康生理信息",
            leaf="个人基本概况信息", raw_level="4", row=9,
        ),
    ]
    standard, report = build_finance_standard(
        entries, source_file="f", source_sheet="Table 1"
    )
    assert len(standard.entries) == 2
    assert standard.trainable_category_count() == 1
    entry_ids = {e.standard_entry_id for e in standard.entries}
    assert entry_ids == {
        "finance:客户.个人.个人自然信息.个人基本概况信息",
        "finance:客户.个人.个人健康生理信息.个人基本概况信息",
    }
    paths = {e.path for e in standard.entries}
    assert ("客户", "个人", "个人自然信息", "个人基本概况信息") in paths
    assert ("客户", "个人", "个人健康生理信息", "个人基本概况信息") in paths
    # distinct levels are preserved per entry (no level-conflict collapse)
    levels = {e.standard_data_level for e in standard.entries}
    assert levels == {"L3", "L4"}
    assert report.issues == []


def test_finance_unparseable_levels_reported_not_fixed():
    entries = [
        _entry(level_1="经营管理", level_2="综合管理", level_3="", leaf="市场营销信息（非公开）", raw_level="l", row=150),
        _entry(level_1="客户", level_2="个人", level_3="个人基本概况", leaf="个人健康生理影像信息", raw_level="3 4", row=168),
    ]
    standard, report = build_finance_standard(
        entries, source_file="f", source_sheet="Table 1"
    )
    for entry in standard.entries:
        assert entry.standard_data_level is None
        assert entry.raw_level in ("l", "3 4")
    kinds = {i.kind for i in report.issues}
    assert "level_unparseable" in kinds
    assert len(report.issues) == 2


def test_finance_build_deterministic_under_input_shuffle():
    entries = [
        _entry(level_1="客户", level_2="个人", level_3="个人自然信息", leaf="个人基本概况信息", raw_level="3", row=3),
        _entry(level_1="业务", level_2="账户信息", level_3="", leaf="基本信息", raw_level="2", row=40),
        _entry(level_1="经营管理", level_2="技术管理", level_3="系统管理信息", leaf="配置信息", raw_level="l", row=99),
    ]
    a, _ = build_finance_standard(list(entries), source_file="f", source_sheet="Table 1")
    b, _ = build_finance_standard(list(reversed(entries)), source_file="f", source_sheet="Table 1")
    assert a.fingerprint() == b.fingerprint()
    assert [e.standard_entry_id for e in a.entries] == [e.standard_entry_id for e in b.entries]
    assert a.to_mapping()["entries"] == b.to_mapping()["entries"]


def test_finance_deterministic_with_duplicate_category_id_entries():
    # Same training category, five distinct 三级 entries: order must not matter
    # (the fact layer keeps every entry, so "first seen" decides nothing).
    entries_a = [
        _entry(level_1="业务", level_2="合约协议", level_3=l3, leaf="基本信息", raw_level="2", row=row)
        for l3, row in [("合同通用信息", 56), ("贷款业务信息", 57), ("中间业务信息", 67), ("资金业务信息", 74), ("其他支付业务信息", 79)]
    ]
    entries_b = list(reversed(entries_a))
    a, airep = build_finance_standard(entries_a, source_file="f", source_sheet="Table 1")
    b, brep = build_finance_standard(entries_b, source_file="f", source_sheet="Table 1")
    assert len(a.entries) == 5 == len(b.entries)
    assert a.trainable_category_count() == 1
    assert a.fingerprint() == b.fingerprint()
    # all five paths are preserved (no first-wins collapse)
    paths = {e.path[-2] for e in a.entries}
    assert paths == {"合同通用信息", "贷款业务信息", "中间业务信息", "资金业务信息", "其他支付业务信息"}
    assert airep.standard_entries_out == 5


def test_reader_issues_merged_into_build_report():
    entries = [_entry(level_1="客户", level_2="个人", level_3="个人自然信息", leaf="个人基本概况信息", raw_level="3", row=3)]
    _, report = build_finance_standard(
        entries, source_file="f", source_sheet="Table 1",
        reader_issues=["finance row 999: too few columns"],
    )
    issues = report.to_mapping()["issues"]
    assert any(i["kind"] == "reader_issue" and "row 999" in i["detail"] for i in issues)
