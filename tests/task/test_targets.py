"""Contract tests for the canonical target / corpus / registry design (stage 2)."""

from __future__ import annotations

import pytest

from agent.task import (
    BUILTIN_DATASET_CONFIGS,
    ClassificationTargetResolver,
    CorpusCategory,
    DatasetConfig,
    LeafRegistry,
    SampleTarget,
    code_leaf_map,
    leaf_registry_from_corpus,
    qualified_category_id,
)


def _record(classification: dict[str, str]) -> dict:
    return {
        "id": "x",
        "classification": classification,
        "label_status": "labeled",
        "metadata": {},
    }


# --- 1. same leaf name under different parents never collides -----------------


def test_same_leaf_different_parents_no_category_id_collision() -> None:
    config = BUILTIN_DATASET_CONFIGS["finance"]
    resolver = ClassificationTargetResolver(config)
    target_a = resolver.resolve(
        _record(
            {
                "level_1": "业务",
                "level_2": "账户信息",
                "level_3": "",
                "level_4": "交易清结算信息",
            }
        )
    )
    target_b = resolver.resolve(
        _record(
            {
                "level_1": "业务",
                "level_2": "交易信息",
                "level_3": "交易通用信息",
                "level_4": "交易清结算信息",
            }
        )
    )
    assert target_a is not None and target_b is not None
    assert target_a.leaf_name == target_b.leaf_name == "交易清结算信息"
    assert target_a.category_id != target_b.category_id
    assert target_a.category_path != target_b.category_path
    assert ("",) not in target_a.category_path  # empty levels omitted from path
    assert "交易清结算信息" in target_a.category_path


def test_path_category_id_is_human_readable_and_path_qualified() -> None:
    config = BUILTIN_DATASET_CONFIGS["finance"]
    resolver = ClassificationTargetResolver(config)
    target = resolver.resolve(
        _record(
            {
                "level_1": "业务",
                "level_2": "账户信息",
                "level_3": "",
                "level_4": "配置信息",
            }
        )
    )
    assert target is not None
    # readable: contains every path level; qualified: fixed 4 slots, empty
    # level_3 is kept as an empty slot so it cannot collide with a 3-level path
    assert target.category_id == "finance:业务.账户信息..配置信息"
    assert not target.category_id.startswith("finance:" + "0" * 16)


def test_same_leaf_same_parents_same_id() -> None:
    config = BUILTIN_DATASET_CONFIGS["finance"]
    resolver = ClassificationTargetResolver(config)
    classification = {
        "level_1": "业务",
        "level_2": "账户信息",
        "level_3": "",
        "level_4": "配置信息",
    }
    first = resolver.resolve(_record(classification))
    second = resolver.resolve(_record(classification))
    assert first is not None and second is not None
    assert first.category_id == second.category_id


# --- 2. shougang code identity is stable --------------------------------------


def test_shougang_code_category_stable_id() -> None:
    config = BUILTIN_DATASET_CONFIGS["shougang"]
    code_map = {
        "科研设备预约管理": "A1-1-1",
        "合同归并": "B1-2",
    }
    resolver = ClassificationTargetResolver(config, code_leaf_map=code_map)
    classification = {
        "level_1": "研发数据域",
        "level_2": "科研管理",
        "level_3": "设备管理",
        "level_4": "科研设备预约管理",
    }
    first = resolver.resolve(_record(classification))
    second = resolver.resolve(_record(classification))
    assert first is not None and second is not None
    assert first.category_id == "A1-1-1"
    assert first.category_id == second.category_id  # stable across calls
    assert first.category_path == (
        "研发数据域",
        "科研管理",
        "设备管理",
        "科研设备预约管理",
    )


def test_shougang_placeholder_resolves_to_none() -> None:
    config = BUILTIN_DATASET_CONFIGS["shougang"]
    resolver = ClassificationTargetResolver(config, code_leaf_map={})
    target = resolver.resolve(
        _record(
            {
                "level_1": "生产数据域",
                "level_2": "产出管理",
                "level_3": "实重管理",
                "level_4": "——",
            }
        )
    )
    assert target is None
    assert resolver.skipped == 1
    assert resolver.resolved == 0
    assert resolver.unresolved == {}


# --- 2b. code strategy never silently falls back ------------------------------


def test_code_strategy_missing_code_is_unresolved_not_fallback() -> None:
    config = BUILTIN_DATASET_CONFIGS["shougang"]
    # corpus incomplete: leaf has no code entry
    resolver = ClassificationTargetResolver(config, code_leaf_map={})
    target = resolver.resolve(
        _record(
            {
                "level_1": "研发数据域",
                "level_2": "科研管理",
                "level_3": "设备管理",
                "level_4": "科研设备预约管理",
            }
        )
    )
    assert target is None  # never remapped to a path-based ID
    assert resolver.resolved == 0
    assert resolver.skipped == 0
    assert resolver.unresolved == {"科研设备预约管理": 1}


# --- 3. pers_info works without any corpus ------------------------------------


def test_pers_info_without_corpus_still_generates_target() -> None:
    config = BUILTIN_DATASET_CONFIGS["pers_info"]
    resolver = ClassificationTargetResolver(config)  # no code map at all
    target = resolver.resolve(
        _record(
            {
                "level_1": "",
                "level_2": "",
                "level_3": "",
                "level_4": "学籍管理信息",
            }
        )
    )
    assert target is not None
    assert target.leaf_level == "level_4"
    assert target.leaf_name == "学籍管理信息"
    assert target.category_path == ("学籍管理信息",)
    assert target.category_id == "pers_info:学籍管理信息"


# --- 4. category_id deterministic ---------------------------------------------


def test_qualified_category_id_is_deterministic_and_whitespace_normalized() -> None:
    a = qualified_category_id("finance", ["业务", "账户信息", "", "配置信息"])
    b = qualified_category_id("finance", ["业务", "账户信息", "", "配置信息"])
    assert a == b
    spaced = qualified_category_id("finance", ["经营 管理", "技术管理", "系统管理信息", "配置信息"])
    compacted = qualified_category_id("finance", ["经营管理", "技术管理", "系统管理信息", "配置信息"])
    assert spaced == compacted
    assert a != spaced
    # human-readable: domain prefix, dot-separated path parts, no hash digest
    assert a.startswith("finance:")
    assert len(a.split(":")) == 2
    assert "." in a


def test_category_id_does_not_depend_on_random_uuid() -> None:
    config = BUILTIN_DATASET_CONFIGS["pers_info"]
    classification = {
        "level_1": "",
        "level_2": "",
        "level_3": "",
        "level_4": "课程信息",
    }
    ids = {
        ClassificationTargetResolver(config).resolve(_record(classification)).category_id
        for _ in range(3)
    }
    assert len(ids) == 1


# --- 5. LeafRegistry rejects duplicate category_id -----------------------------


def test_leaf_registry_rejects_duplicate_category_id() -> None:
    from agent.task import LeafCategory

    categories = tuple(
        LeafCategory(category_id=category_id, name="n")
        for category_id in ("A1-1-1", "A1-1-2", "A1-1-3", "A1-1-4", "B1-1-1", "A1-1-1")
    )
    with pytest.raises(ValueError, match="unique"):
        LeafRegistry(categories)


def test_leaf_registry_from_mapping_rejects_duplicate_category_id() -> None:
    with pytest.raises(ValueError, match="unique"):
        LeafRegistry.from_mapping(
            {
                "categories": [
                    {"category_id": "A1-1-1", "name": "科研设备预约管理"},
                    {"category_id": "A1-1-2", "name": "科研进程管控"},
                    {"category_id": "A1-1-3", "name": "非常规委托检测管理"},
                    {"category_id": "A1-1-4", "name": "考试管理"},
                    {"category_id": "B1-1-1", "name": "薄板合同产线专业化设计"},
                    {"category_id": "A1-1-1", "name": "duplicate"},
                ]
            }
        )


def test_leaf_registry_roundtrip_keeps_new_fields() -> None:
    raw = {
        "categories": [
            {
                "category_id": "A1-1-1",
                "name": "科研设备预约管理",
                "description": "设备预约信息",
                "path": ["研发数据域", "科研设备管理"],
                "code": "A1-1-1",
            },
            {"category_id": "A1-1-2", "name": "科研进程管控", "path": ["研发数据域"]},
            {"category_id": "A1-1-3", "name": "非常规委托检测管理"},
            {"category_id": "A1-1-4"},
            {"category_id": "B1-1-1", "name": "薄板合同产线专业化设计"},
        ]
    }
    registry = LeafRegistry.from_mapping(raw)
    category = registry.get("A1-1-1")
    assert category.name == "科研设备预约管理"
    assert category.path == ("研发数据域", "科研设备管理")
    assert category.code == "A1-1-1"
    # old-style entries without name fall back to category_id as display name
    assert registry.get("A1-1-4").name == "A1-1-4"


# --- 5b. registry represents the corpus leaf universe, not training samples ----


def test_leaf_registry_from_corpus_covers_universe_including_unseen_leaves() -> None:
    corpus = [
        CorpusCategory(category_id="A1-1-1", name="科研设备预约管理", description="设备预约信息"),
        CorpusCategory(category_id="A1-1-2", name="科研进程管控"),
        CorpusCategory(category_id="A1-1-3", name="非常规委托检测管理"),
        CorpusCategory(category_id="A1-1-4", name="考试管理"),
        CorpusCategory(category_id="B1-1-1", name="薄板合同产线专业化设计", code="B1-1-1"),
        # a leaf that never appears in any training sample must still be in
        # the registry when the standard defines it
        CorpusCategory(category_id="B1-1-2", name="厚板组板设计"),
    ]
    registry = leaf_registry_from_corpus(corpus)
    assert isinstance(registry, LeafRegistry)
    assert set(registry.ids) == {
        "A1-1-1",
        "A1-1-2",
        "A1-1-3",
        "A1-1-4",
        "B1-1-1",
        "B1-1-2",
    }
    assert registry.get("B1-1-2").description == ""  # missing description is legal
    assert registry.get("B1-1-1").code == "B1-1-1"


def test_leaf_registry_from_corpus_rejects_duplicate_ids() -> None:
    corpus = [
        CorpusCategory(category_id="A1-1-1", name="科研设备预约管理"),
        CorpusCategory(category_id="A1-1-1", name="科研进程管控"),
        CorpusCategory(category_id="A1-1-2", name="非常规委托检测管理"),
        CorpusCategory(category_id="A1-1-3", name="考试管理"),
        CorpusCategory(category_id="A1-1-4", name="薄板合同产线专业化设计"),
        CorpusCategory(category_id="B1-1-1", name="厚板组板设计"),
    ]
    with pytest.raises(ValueError, match="unique"):
        leaf_registry_from_corpus(corpus)


# --- code_leaf_map ambiguity guard --------------------------------------------


def test_code_leaf_map_rejects_ambiguous_leaf_to_code() -> None:
    categories = [
        CorpusCategory(category_id="a", name="基本信息", code="A1-1"),
        CorpusCategory(category_id="b", name="基本信息", code="A3-1"),
    ]
    with pytest.raises(ValueError, match="multiple codes"):
        code_leaf_map(categories)


def test_code_leaf_map_skips_categories_without_code() -> None:
    categories = [
        CorpusCategory(category_id="a", name="学籍管理信息"),
        CorpusCategory(category_id="b", name="科研设备预约管理", code="A1-1-1"),
    ]
    assert code_leaf_map(categories) == {"科研设备预约管理": "A1-1-1"}


# --- infra shares the shougang registry ---------------------------------------


def test_infra_reuses_shougang_registry_source() -> None:
    config = BUILTIN_DATASET_CONFIGS["infra"]
    assert config.registry_source == "shougang"
    assert config.id_strategy == "code"


def test_dataset_config_validation() -> None:
    with pytest.raises(ValueError, match="id_strategy"):
        DatasetConfig(dataset="x", id_strategy="semantic")
    with pytest.raises(ValueError, match="leaf_level"):
        DatasetConfig(dataset="x", leaf_level="level_9")
    with pytest.raises(ValueError, match="registry_source"):
        DatasetConfig(dataset="x", registry_source="x")


# --- corpus contract allows missing description / multiple examples -----------


def test_corpus_category_allows_missing_description_and_examples() -> None:
    category = CorpusCategory(
        category_id="pers:abc",
        name="学籍管理信息",
    )
    assert category.description == ""
    assert category.examples == ()
    assert category.code is None
    assert category.path == ()

    rich = CorpusCategory(
        category_id="A1-1-1",
        name="科研设备预约管理",
        description="描述",
        path=("研发数据域", "科研设备管理"),
        code="A1-1-1",
        examples=("设备预约信息", "审核表"),
    )
    assert len(rich.examples) == 2

    with pytest.raises(ValueError, match="category_id"):
        CorpusCategory(category_id="  ", name="x")
    with pytest.raises(ValueError, match="name"):
        CorpusCategory(category_id="x", name=" ")
