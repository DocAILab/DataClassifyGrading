"""Phase-2 tests: LeafRegistry/Corpus derived from CanonicalStandard.

Hermetic (synthetic CanonicalStandard fixtures) plus real-data parity tests
that skip when data/standards (regenerable) is absent.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.standards.contracts import CanonicalStandard, SourceRef, StandardCategory
from agent.task.canonical_corpus import (
    BuildReport,
    build_from_standard,
    build_pers_info_corpus,
    corpus_to_mapping,
    registry_to_mapping,
)
from agent.task.canonical_dataset import load_corpus_categories

ROOT = Path(__file__).resolve().parents[2]
STANDARDS = ROOT / "data" / "standards"
REG = ROOT / "cfg" / "task" / "registry"


def _entry(
    entry_id: str,
    category_id: str,
    name: str,
    row: int,
    desc: str = "",
    level: str = "L2",
    path: tuple = (),
    code: str | None = None,
) -> StandardCategory:
    return StandardCategory(
        standard_entry_id=entry_id,
        category_id=category_id,
        name=name,
        path=path,
        description=desc,
        standard_data_level=level,
        raw_level=level,
        code=code,
        source=SourceRef(file="f.xlsx", sheet="s", row=row),
    )


FIVE = [
    _entry(
        "finance:业务.合约协议.合同通用信息.基本信息", "finance:业务.合约协议.基本信息",
        "基本信息", 56, "合同通用定义", path=("业务", "合约协议", "合同通用信息", "基本信息"),
    ),
    _entry(
        "finance:业务.合约协议.贷款业务信息.基本信息", "finance:业务.合约协议.基本信息",
        "基本信息", 57, "贷款业务定义", path=("业务", "合约协议", "贷款业务信息", "基本信息"),
    ),
    _entry(
        "finance:业务.合约协议.中间业务信息.基本信息", "finance:业务.合约协议.基本信息",
        "基本信息", 67, "中间业务定义", path=("业务", "合约协议", "中间业务信息", "基本信息"),
    ),
    _entry(
        "finance:业务.合约协议.资金业务信息.基本信息", "finance:业务.合约协议.基本信息",
        "基本信息", 74, "资金业务定义", path=("业务", "合约协议", "资金业务信息", "基本信息"),
    ),
    _entry(
        "finance:业务.合约协议.非银行支付业务信息.基本信息", "finance:业务.合约协议.基本信息",
        "基本信息", 79, "非银行支付定义", path=("业务", "合约协议", "非银行支付业务信息", "基本信息"),
    ),
]


def test_finance_projection_keeps_all_entries_lossless():
    standard = CanonicalStandard(
        dataset="finance", id_strategy="path", standard_source=SourceRef(),
        entries=tuple(FIVE),
    )
    categories, report = build_from_standard(
        standard, dataset="finance", excluded_category_ids=set(),
    )
    assert report.categories_out == 1
    assert report.standard_entries_out == 5
    bucket = categories[0]
    assert len(bucket.standard_entries) == 5            # NOT first-only
    assert len(bucket.standard_entry_ids) == 5
    # descriptions keep EVERY entry's text, in Excel-row order
    assert [e.description for e in bucket.standard_entries] == [
        "合同通用定义", "贷款业务定义", "中间业务定义", "资金业务定义", "非银行支付定义"
    ]
    # Stage-2-facing primary description = first source row (matches historical)
    assert bucket.description == "合同通用定义"
    assert bucket.descriptions == ("贷款业务定义", "中间业务定义", "资金业务定义", "非银行支付定义")
    # registry view path keeps Stage-1 values (L1/L2/leaf)
    assert bucket.path == ("业务", "合约协议", "基本信息")
    assert all(e.path for e in bucket.standard_entries)  # real depth kept in entries


def test_corpus_round_trip_preserves_standard_entries():
    standard = CanonicalStandard(
        dataset="finance", id_strategy="path", standard_source=SourceRef(),
        entries=tuple(FIVE),
    )
    categories, _ = build_from_standard(
        standard, dataset="finance", excluded_category_ids=set(),
    )
    payload = corpus_to_mapping(
        "finance", "data/standards/finance.standard.json", "path",
        categories, BuildReport(dataset="finance", source="x", id_strategy="path"),
    )
    rebuilt = load_corpus_categories(
        _write_tmp(payload)
    )
    bucket = rebuilt[0]
    assert [e.standard_entry_id for e in bucket.standard_entries] == [
        e.standard_entry_id for e in categories[0].standard_entries
    ]
    assert [e.description for e in bucket.standard_entries] == [
        e.description for e in categories[0].standard_entries
    ]


def _write_tmp(payload: dict) -> Path:
    import tempfile

    path = Path(tempfile.mkstemp(suffix=".json")[1])
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def test_shougang_explicit_projection_policy_excludes_b36():
    standard = CanonicalStandard(
        dataset="shougang", id_strategy="code", standard_source=SourceRef(),
        entries=(
            _entry("A1-1-1", "A1-1-1", "设备预约", 3, "a", "L2", code="A1-1-1"),
            _entry("B3-6", "B3-6", "中厚板作业计划", 35, "b", "L2", code="B3-6"),
        ),
    )
    categories, report = build_from_standard(
        standard, dataset="shougang",
        excluded_category_ids={"B3-6"},  # explicit projection policy
    )
    assert [c.category_id for c in categories] == ["A1-1-1"]
    assert report.excluded_categories == ["B3-6"]
    assert any(issue.kind == "category_excluded" for issue in report.issues)
    # NOT silently added: B3-6 not in active categories
    assert all(c.category_id != "B3-6" for c in categories)


def test_projection_policy_default_from_dataset_config():
    # the formal builder applies DatasetConfig.projection_excluded_category_ids
    # when no explicit exclusion is passed (infra inherits shougang's policy)
    standard = CanonicalStandard(
        dataset="shougang", id_strategy="code", standard_source=SourceRef(),
        entries=(
            _entry("A1-1-1", "A1-1-1", "设备预约", 3, "a", "L2", code="A1-1-1"),
            _entry("B3-6", "B3-6", "中厚板作业计划", 35, "b", "L2", code="B3-6"),
        ),
    )
    for dataset in ("shougang", "infra"):
        categories, report = build_from_standard(standard, dataset=dataset)
        assert [c.category_id for c in categories] == ["A1-1-1"]
        assert report.excluded_categories == ["B3-6"]
    # finance policy has no exclusions
    finance = CanonicalStandard(
        dataset="finance", id_strategy="path", standard_source=SourceRef(),
        entries=tuple(FIVE),
    )
    categories, report = build_from_standard(finance, dataset="finance")
    assert report.excluded_categories == []
    assert len(categories) == 1


def test_no_exclusion_keeps_all_categories():
    standard = CanonicalStandard(
        dataset="shougang", id_strategy="code", standard_source=SourceRef(),
        entries=(
            _entry("A", "A", "a", 3, "a", "L2", code="A"),
            _entry("B", "B", "b", 4, "b", "L2", code="B"),
        ),
    )
    categories, report = build_from_standard(
        standard, dataset="shougang", excluded_category_ids=set()
    )
    assert len(categories) == 2
    assert report.excluded_categories == []


def test_infra_reuses_shougang_standard():
    standard = CanonicalStandard(
        dataset="shougang", id_strategy="code", standard_source=SourceRef(),
        entries=(_entry("A", "A", "a", 3, "a", "L2", code="A"),),
    )
    categories, report = build_from_standard(
        standard, dataset="infra", registry_source="shougang",
        excluded_category_ids=set(),
    )
    assert report.registry_source == "shougang"
    assert categories[0].path == ()  # shougang Stage-1 registry path stays empty


def test_pers_info_fallback_has_no_standard_facts():
    records = [
        {"id": "1", "label_status": "labeled",
         "classification": {"level_1": "", "level_2": "", "level_3": "", "level_4": "学籍管理信息"}}
    ]
    categories, _ = build_pers_info_corpus(records, "pers_info")
    assert len(categories) == 1
    assert categories[0].standard_entries == ()
    assert categories[0].standard_entry_ids == ()
    assert categories[0].scoped_annotations == ()


def test_build_deterministic_regardless_of_entry_arrival_order():
    entries = list(FIVE) + [_entry("f:x.y.z", "f:x.y.z", "z", 5, "z", "L3", path=("x", "y", "z"))]
    a, _ = build_from_standard(
        CanonicalStandard("finance", "path", SourceRef(), entries=tuple(entries)),
        dataset="finance", excluded_category_ids=set(),
    )
    b, _ = build_from_standard(
        CanonicalStandard("finance", "path", SourceRef(), entries=tuple(reversed(entries))),
        dataset="finance", excluded_category_ids=set(),
    )
    assert [c.category_id for c in a] == [c.category_id for c in b]
    assert [c.standard_entry_ids for c in a] == [c.standard_entry_ids for c in b]


# ---- real-data parity (skip when the regenerable standard layer is absent) ----

_REAL = pytest.mark.skipif(
    not (STANDARDS / "finance.standard.json").is_file()
    or not (STANDARDS / "shougang.standard.json").is_file(),
    reason="canonical standards not present (restore data/raw and run script.standard.cli)",
)


@_REAL
@pytest.mark.parametrize("dataset,key", [("finance", "finance"), ("shougang", "shougang")])
def test_real_registry_identical_to_committed(dataset, key):
    import re

    from agent.standards.contracts import CanonicalStandard

    standard = CanonicalStandard.from_mapping(
        json.load(open(STANDARDS / f"{key}.standard.json", encoding="utf-8"))
    )
    categories, _ = build_from_standard(standard, dataset=dataset)
    built = registry_to_mapping(categories)
    committed = json.load(open(REG / f"{dataset}.registry.json", encoding="utf-8"))
    # Stage-1 prompt surface must be byte-identical: id set, order, name, path,
    # code. Description is NOT used by Stage 1; the only permitted divergence
    # is whitespace-only (the legacy dict silently dropped a stray source space
    # in finance 经营管理/运营管理/档案资料管理信息).
    bo = [c["category_id"] for c in built["categories"]]
    co = [c["category_id"] for c in committed["categories"]]
    assert bo == co
    bm = {c["category_id"]: c for c in built["categories"]}
    cm = {c["category_id"]: c for c in committed["categories"]}
    for category_id in bm:
        assert bm[category_id]["name"] == cm[category_id]["name"]
        assert bm[category_id]["path"] == cm[category_id]["path"]
        assert bm[category_id].get("code") == cm[category_id].get("code")
        a, b = bm[category_id]["description"], cm[category_id]["description"]
        assert re.sub(r"\s+", "", a) == re.sub(r"\s+", "", b)  # whitespace-only ok


@_REAL
def test_real_finance_corpus_preserves_five_entries_and_annotations():
    from agent.standards.contracts import CanonicalStandard

    standard = CanonicalStandard.from_mapping(
        json.load(open(STANDARDS / "finance.standard.json", encoding="utf-8"))
    )
    categories, _ = build_from_standard(standard, dataset="finance")
    bucket = next(c for c in categories if c.category_id == "finance:业务.合约协议.基本信息")
    assert len(bucket.standard_entries) == 5
    # scoped annotations attached to the 金融监管和服务 40-leaf group
    annotated = [c for c in categories if c.scoped_annotations]
    assert len(annotated) == 43  # 40 (J93:J132) + 2 (J168:J169) + 1 (J55)


@_REAL
def test_real_shougang_b36_excluded_and_reported():
    from agent.standards.contracts import CanonicalStandard

    standard = CanonicalStandard.from_mapping(
        json.load(open(STANDARDS / "shougang.standard.json", encoding="utf-8"))
    )
    categories, report = build_from_standard(standard, dataset="shougang")
    assert len(categories) == 233
    assert report.standard_entries_out == 234
    assert report.excluded_categories == ["B3-6"]


def _shougang_entry():
    return StandardCategory(
        standard_entry_id="A1-1-1",
        category_id="A1-1-1",
        name="科研设备预约管理",
        path=("研发数据域", "产品研发", "科研检验", "科研设备预约管理"),
        description="指科研检验设备预约过程中产生的信息",   # 四级定义
        code="A1-1-1",
        standard_data_level="L2",
        raw_level="2",
        content="设备预约信息、审核表",                      # 数据资源说明
        raw_fields={
            "level_1_definition": {"value": "研发域定义", "source_cell": "C3"},
            "resource": {"value": "科研实验室", "source_cell": "L3", "merged_range": None},
        },
        source=SourceRef(file="data/raw/关基-数据分类分级目录.xlsx", sheet="数据分类分级", row=3),
    )


def test_standard_entry_view_round_trip_keeps_content_code_source():
    from agent.task.contracts import StandardEntryView

    standard = CanonicalStandard(
        dataset="shougang", id_strategy="code", standard_source=SourceRef(),
        entries=(_shougang_entry(),),
    )
    categories, _ = build_from_standard(
        standard, dataset="shougang", excluded_category_ids=set()
    )
    view = categories[0].standard_entries[0]
    assert view.content == "设备预约信息、审核表"
    assert view.code == "A1-1-1"
    assert view.source == {"file": "data/raw/关基-数据分类分级目录.xlsx",
                           "sheet": "数据分类分级", "row": 3}
    # corpus JSON -> load_corpus_categories() full round-trip, no field loss
    payload = corpus_to_mapping(
        "shougang", "x.standard.json", "code", categories,
        BuildReport(dataset="shougang", source="x", id_strategy="code"),
    )
    rebuilt = load_corpus_categories(_write_tmp(payload))
    view2 = rebuilt[0].standard_entries[0]
    assert view2.content == view.content
    assert view2.code == view.code
    assert view2.source == view.source
    assert view2.raw_fields == view.raw_fields
    assert view2.standard_data_level == view.standard_data_level
    assert view2.path == view.path


def test_shougang_description_content_resource_kept_separately():
    # description=四级定义, content=数据资源说明, resource(数据来源) stays in
    # raw_fields — the three must never be confused or lost.
    standard = CanonicalStandard(
        dataset="shougang", id_strategy="code", standard_source=SourceRef(),
        entries=(_shougang_entry(),),
    )
    categories, _ = build_from_standard(
        standard, dataset="shougang", excluded_category_ids=set()
    )
    category = categories[0]
    view = category.standard_entries[0]
    assert category.description == "指科研检验设备预约过程中产生的信息"  # 四级定义
    assert view.description == "指科研检验设备预约过程中产生的信息"
    assert view.content == "设备预约信息、审核表"                          # 数据资源说明
    assert view.raw_fields["resource"]["value"] == "科研实验室"            # 数据来源
    assert "resource" not in (view.description or "") and view.description != view.content
