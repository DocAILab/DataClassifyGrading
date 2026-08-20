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
        standard, dataset="finance",
        previous_active_ids={"finance:业务.合约协议.基本信息"},
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
        standard, dataset="finance",
        previous_active_ids={"finance:业务.合约协议.基本信息"},
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


def test_shougang_standard_minus_previous_active_reported():
    standard = CanonicalStandard(
        dataset="shougang", id_strategy="code", standard_source=SourceRef(),
        entries=(
            _entry("A1-1-1", "A1-1-1", "设备预约", 3, "a", "L2", code="A1-1-1"),
            _entry("B3-6", "B3-6", "中厚板作业计划", 35, "b", "L2", code="B3-6"),
        ),
    )
    categories, report = build_from_standard(
        standard, dataset="shougang",
        previous_active_ids={"A1-1-1"},  # B3-6 absent from the old registry
    )
    assert [c.category_id for c in categories] == ["A1-1-1"]
    assert report.excluded_categories == ["B3-6"]
    assert any(issue.kind == "category_excluded" for issue in report.issues)
    # NOT silently added: B3-6 not in active categories
    assert all(c.category_id != "B3-6" for c in categories)


def test_no_previous_active_keeps_all_categories():
    standard = CanonicalStandard(
        dataset="shougang", id_strategy="code", standard_source=SourceRef(),
        entries=(
            _entry("A", "A", "a", 3, "a", "L2", code="A"),
            _entry("B", "B", "b", 4, "b", "L2", code="B"),
        ),
    )
    categories, report = build_from_standard(
        standard, dataset="shougang", previous_active_ids=None
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
        previous_active_ids={"A"},
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
        dataset="finance", previous_active_ids=None,
    )
    b, _ = build_from_standard(
        CanonicalStandard("finance", "path", SourceRef(), entries=tuple(reversed(entries))),
        dataset="finance", previous_active_ids=None,
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
    committed_ids = {
        str(c.get("category_id"))
        for c in json.load(open(REG / f"{dataset}.registry.json", encoding="utf-8"))["categories"]
    }
    categories, _ = build_from_standard(
        standard, dataset=dataset, previous_active_ids=committed_ids
    )
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
    committed_ids = {
        str(c.get("category_id"))
        for c in json.load(open(REG / "finance.registry.json", encoding="utf-8"))["categories"]
    }
    categories, _ = build_from_standard(
        standard, dataset="finance", previous_active_ids=committed_ids
    )
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
    committed_ids = {
        str(c.get("category_id"))
        for c in json.load(open(REG / "shougang.registry.json", encoding="utf-8"))["categories"]
    }
    categories, report = build_from_standard(
        standard, dataset="shougang", previous_active_ids=committed_ids
    )
    assert len(categories) == 233
    assert report.standard_entries_out == 234
    assert report.excluded_categories == ["B3-6"]
