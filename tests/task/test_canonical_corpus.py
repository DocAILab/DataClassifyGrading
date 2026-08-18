"""Tests for the stage-3A canonical corpus/standard -> CorpusCategory -> LeafRegistry
conversion. Guanji / financial standard tests read the real files under
data/knowledge/standards_map (checked into the repo, available in CI);
pers_info tests use an inline dataset-universe fixture."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.task import BUILTIN_DATASET_CONFIGS, LeafRegistry, leaf_registry_from_corpus
from agent.task.canonical_corpus import (
    aggregate_categories,
    build_finance_corpus,
    build_pers_info_corpus,
    build_shougang_corpus,
    parse_financial_standard,
    parse_guanji_standard,
    pers_info_categories,
    registry_to_mapping,
)

STANDARDS = Path(__file__).resolve().parents[2] / "data" / "knowledge" / "standards_map"
GUANJI_PATH = STANDARDS / "guanji_dict.json"
FINANCIAL_PATH = STANDARDS / "financial_standards_dict.json"

PERS_INFO_UNIVERSE = [
    "人力资源数据", "任课信息", "基本信息年级信息和班级信息", "学历学位信息",
    "学校概况基本信息", "学生个人基本信息", "学生敏感个人信息", "学科类科研管理",
    "学籍基本数据", "学籍管理信息", "教职工个人基本信息", "毕业就业信息",
    "离退休信息", "系统数据", "考核信息", "考试成绩", "职称信息", "课程信息",
]


def _pers_records(leaves: list[str]) -> list[dict]:
    return [
        {
            "id": f"p{i}",
            "classification": {"level_1": "", "level_2": "", "level_3": "", "level_4": leaf},
        }
        for i, leaf in enumerate(leaves)
    ]


# --- guanji (shougang / infra) ------------------------------------------------


def test_guanji_parse_full_universe_and_malformed_skip() -> None:
    categories, issues = parse_guanji_standard(json_load(GUANJI_PATH), dataset="shougang")
    assert len(categories) == 233  # 234 entries - 1 malformed 'nan' entry
    assert any(issue.kind == "malformed_skipped" for issue in issues)
    # every category keeps its opaque code as identity
    assert all(category.code for category in categories)
    assert len({category.category_id for category in categories}) == 233
    # sample: name extracted, code kept, description from the mapping key
    first = next(c for c in categories if c.category_id == "A1-1-1")
    assert first.name == "科研设备预约管理"
    assert first.description  # description comes from the standard's key
    # a short two-group code exists (B1-2 合同归并)
    assert any(c.category_id == "B1-2" and c.name == "合同归并" for c in categories)


def test_shougang_corpus_build_and_registry_roundtrip() -> None:
    categories, report = build_shougang_corpus(GUANJI_PATH, dataset="shougang")
    assert report.entries_read == 234
    assert report.categories_out == 233
    assert report.id_strategy == "code"
    assert report.aggregated == {"kinds": 0, "instances": 0}

    registry = leaf_registry_from_corpus(categories)
    assert isinstance(registry, LeafRegistry)
    assert len(registry.ids) == 233
    # registry JSON roundtrip: from_mapping must accept the serialized form
    registry2 = LeafRegistry.from_mapping(registry_to_mapping(categories))
    assert set(registry2.ids) == set(registry.ids)


def test_infra_uses_shougang_universe() -> None:
    categories, _ = build_shougang_corpus(GUANJI_PATH, dataset="infra")
    assert len(categories) == 233
    assert BUILTIN_DATASET_CONFIGS["infra"].registry_source == "shougang"
    # infra's 4 dataset leaves are covered by the shared universe
    infra_leaves = {"厚板组板设计", "水质监测管理", "薄板合同产线专业化设计", "非常规委托检测管理"}
    names = {category.name for category in categories}
    assert infra_leaves <= names


# --- financial standard (finance) ---------------------------------------------


def test_financial_standard_parse_path_qualified_ids() -> None:
    categories, issues = parse_financial_standard(json_load(FINANCIAL_PATH), dataset="finance")
    assert len(categories) == 237
    assert issues == []
    by_id = {category.category_id: category for category in categories}
    # 3-segment standard path -> 4-slot ID with empty level_3 slot
    category = by_id["finance:客户.个人..个人基本概况信息"]
    assert category.name == "个人基本概况信息"
    assert category.path == ("客户", "个人", "个人基本概况信息")  # provenance keeps 3 segments
    assert category.code is None
    assert category.description  # description from the mapping key
    # entries outside the dataset vocabulary stay in the universe
    assert any(category_id.startswith("finance:监管.") for category_id in by_id)


def test_financial_standard_duplicate_aggregation() -> None:
    categories, report = build_finance_corpus(FINANCIAL_PATH, dataset="finance")
    assert report.entries_read == 237
    assert report.categories_out == 233
    # one duplicated category (业务-合约协议-基本信息 x5) -> 1 kind / 4 extra entries
    assert report.aggregated == {"kinds": 1, "instances": 4}
    merged = next(c for c in categories if c.category_id == "finance:业务.合约协议..基本信息")
    assert len(merged.examples) == 4  # 4 extra descriptions aggregated
    assert merged.description


def test_financial_registry_id_space_differs_from_4level_dataset_paths() -> None:
    # The standard has no level_3; IDs use an empty level_3 slot. A dataset
    # record with a populated level_3 therefore resolves to an ID that is NOT
    # in this registry (reported as coverage gap, never auto-repaired).
    categories, _ = build_finance_corpus(FINANCIAL_PATH, dataset="finance")
    registry_ids = {category.category_id for category in categories}
    assert "finance:业务.账户信息..基本信息" in registry_ids
    assert "finance:业务.账户信息.系统管理信息.基本信息" not in registry_ids


# --- pers_info ----------------------------------------------------------------


def test_pers_info_categories_from_dataset_universe() -> None:
    categories, issues = pers_info_categories(_pers_records(PERS_INFO_UNIVERSE))
    assert len(categories) == 18
    assert issues == []
    by_name = {category.name: category for category in categories}
    category = by_name["学籍管理信息"]
    assert category.category_id == "pers_info:学籍管理信息"
    assert category.path == ("学籍管理信息",)
    assert category.description == ""  # missing description is allowed
    assert category.code is None


def test_pers_info_build_report_marks_dataset_derivation() -> None:
    categories, report = build_pers_info_corpus(_pers_records(PERS_INFO_UNIVERSE))
    assert report.categories_out == 18
    assert any(issue.kind == "derived_from_dataset_universe" for issue in report.issues)
    registry = leaf_registry_from_corpus(categories)
    assert len(registry.ids) == 18
    assert set(registry.ids) == {
        f"pers_info:{leaf}" for leaf in PERS_INFO_UNIVERSE
    }


def test_pers_info_without_corpus_resolver_targets_match_registry() -> None:
    from agent.task.resolver import ClassificationTargetResolver

    categories, _ = build_pers_info_corpus(_pers_records(PERS_INFO_UNIVERSE))
    registry = leaf_registry_from_corpus(categories)
    resolver = ClassificationTargetResolver(BUILTIN_DATASET_CONFIGS["pers_info"])
    for record in _pers_records(PERS_INFO_UNIVERSE):
        target = resolver.resolve(record)
        assert target is not None
        assert target.category_id in registry.ids


# --- aggregation unit ---------------------------------------------------------


def test_aggregate_categories_merges_descriptions_into_examples() -> None:
    from agent.task import CorpusCategory

    categories = [
        CorpusCategory(category_id="id1", name="a", description="first"),
        CorpusCategory(category_id="id1", name="a", description="second"),
        CorpusCategory(category_id="id1", name="a", description="third"),
        CorpusCategory(category_id="id2", name="b", description="solo"),
    ]
    merged, aggregated = aggregate_categories(categories)
    assert aggregated == {"kinds": 1, "instances": 2}
    assert len(merged) == 2
    by_id = {category.category_id: category for category in merged}
    assert by_id["id1"].description == "first"
    assert by_id["id1"].examples == ("second", "third")
    assert by_id["id2"].examples == ()


def test_aggregate_categories_is_deterministic() -> None:
    categories, report_a = build_finance_corpus(FINANCIAL_PATH, dataset="finance")
    categories_b, report_b = build_finance_corpus(FINANCIAL_PATH, dataset="finance")
    assert report_a.to_mapping() == report_b.to_mapping()
    assert [c.category_id for c in categories] == [c.category_id for c in categories_b]


def json_load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


@pytest.fixture(autouse=True)
def _require_standards_files() -> None:
    if not GUANJI_PATH.is_file() or not FINANCIAL_PATH.is_file():
        pytest.skip("standards_map files not available (not checked out)")
