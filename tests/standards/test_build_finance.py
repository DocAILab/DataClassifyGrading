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
    by_id = standard.by_id()
    full = by_id["finance:客户.个人.个人基本概况信息"]
    assert full.path == ("客户", "个人", "个人自然信息", "个人基本概况信息")  # 4 real levels
    assert full.standard_data_level == "L3"
    assert full.description == "指个人基本情况数据"
    shallow = by_id["finance:业务.账户信息.基本信息"]
    assert shallow.path == ("业务", "账户信息", "基本信息")  # empty 三级 omitted, no padding
    assert shallow.standard_data_level == "L2"
    assert report.issues == []


def test_finance_identity_excludes_level_3_but_path_keeps_it():
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
    # level_3 is provenance only: both rows share the same L1-L2-leaf identity
    assert len(standard.categories) == 1
    category = standard.categories[0]
    assert category.category_id == "finance:客户.个人.个人基本概况信息"
    assert category.path == ("客户", "个人", "个人自然信息", "个人基本概况信息")
    assert report.aggregated == {"kinds": 1, "instances": 1}


def test_finance_unparseable_levels_reported_not_fixed():
    entries = [
        _entry(level_1="经营管理", level_2="综合管理", level_3="", leaf="市场营销信息（非公开）", raw_level="l", row=150),
        _entry(level_1="客户", level_2="个人", level_3="个人基本概况", leaf="个人健康生理影像信息", raw_level="3 4", row=168),
    ]
    standard, report = build_finance_standard(
        entries, source_file="f", source_sheet="Table 1"
    )
    for category in standard.categories:
        assert category.standard_data_level is None
        assert category.raw_level in ("l", "3 4")
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
    assert [c.category_id for c in a.categories] == [c.category_id for c in b.categories]
    assert a.to_mapping()["categories"] == b.to_mapping()["categories"]
