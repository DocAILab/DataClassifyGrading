"""Phase 1 canonical standard — shougang build tests (dict fixtures, hermetic)."""

from __future__ import annotations

from agent.standards.build import build_shougang_standard


def _entry(**kw):
    base = dict(sheet="数据分类分级", row=1, content="", raw_level="", resource="")
    base.update(kw)
    return base


def test_shougang_code_name_path_level():
    entries = [
        _entry(
            level_1="研发数据域（A）", level_2="产品研发（A1）", level_3="科研检验（A1-1）",
            leaf="科研设备预约管理（A1-1-1）", description="指设备预约",
            content="设备预约信息", raw_level="2", row=7,
        )
    ]
    standard, report = build_shougang_standard(
        entries, source_file="data/raw/c.xlsx", source_sheet="数据分类分级"
    )
    category = standard.categories[0]
    assert category.category_id == "A1-1-1"
    assert category.code == "A1-1-1"
    assert category.name == "科研设备预约管理"
    assert category.path == ("研发数据域", "产品研发", "科研检验", "科研设备预约管理")
    assert category.description == "指设备预约"
    assert category.content == "设备预约信息"
    assert category.standard_data_level == "L2"
    assert report.issues == []


def test_shougang_leaf_at_level_three_no_invented_fourth_level():
    # catalog encodes 三级-level leaves with a literal "——" in the 四级 cell
    entries = [
        _entry(
            level_1="生产数据域（B）", level_2="生产合同（订单）（B1）",
            level_3="合同归并（B1-2）", leaf="——",
            description="指按照合同加工途径", raw_level="3", row=9,
        ),
        _entry(
            level_1="管理数据域（C）", level_2="生产质量管理（C1）",
            level_3="质保书管理（C1-5）", leaf="——", raw_level="1", row=108,
        ),
    ]
    standard, report = build_shougang_standard(
        entries, source_file="c", source_sheet="数据分类分级"
    )
    by_id = standard.by_id()
    merged = by_id["B1-2"]
    assert merged.name == "合同归并"
    # real hierarchy depth is 3; the "——" marker does NOT become a 4th level
    assert merged.path == ("生产数据域", "生产合同（订单）", "合同归并")
    assert merged.standard_data_level == "L3"
    assert by_id["C1-5"].standard_data_level == "L1"


def test_shougang_no_code_real_leaf_reported_and_skipped():
    entries = [
        _entry(level_1="管理数据域（C）", level_2="经营管理（C8）", level_3="综合管理",
               leaf="无法归类的真实名称", raw_level="2", row=90),
    ]
    standard, report = build_shougang_standard(
        entries, source_file="c", source_sheet="数据分类分级"
    )
    assert len(standard.categories) == 0
    assert any(issue.kind == "no_code" for issue in report.issues)


def test_shougang_build_deterministic_under_input_shuffle():
    entries = [
        _entry(level_1="研发数据域（A）", level_2="产品研发（A1）", level_3="科研检验（A1-1）", leaf="科研设备预约管理（A1-1-1）", raw_level="2", row=7),
        _entry(level_1="生产数据域（B）", level_2="生产合同（订单）（B1）", level_3="合同归并（B1-2）", leaf="——", raw_level="3", row=9),
        _entry(level_1="管理数据域（C）", level_2="生产质量管理（C1）", level_3="检化验管理（C1-4）", leaf="认证管理（C1-4-4）", raw_level="3", row=105),
        _entry(level_1="管理数据域（C）", level_2="经营管理（C8）", level_3="无码", leaf="无码类别", raw_level="2", row=90),
    ]
    a, _ = build_shougang_standard(list(entries), source_file="c", source_sheet="数据分类分级")
    b, _ = build_shougang_standard(list(reversed(entries)), source_file="c", source_sheet="数据分类分级")
    assert a.fingerprint() == b.fingerprint()
    assert [c.category_id for c in a.categories] == [c.category_id for c in b.categories]
