"""PromptChoiceRegistry: deterministic choice ids and shortest unique suffixes."""

import pytest

from agent.task import (
    LeafCategory,
    LeafRegistry,
    PromptChoice,
    PromptChoiceError,
    PromptChoiceRegistry,
    decode_stage2_answer,
    encode_stage2_answer,
)


def _registry(categories: list[dict]) -> LeafRegistry:
    return LeafRegistry.from_mapping({"categories": categories})


BASIC = [
    {"category_id": "A", "name": "alpha data"},
    {"category_id": "B", "name": "beta data"},
    {"category_id": "C", "name": "gamma data"},
    {"category_id": "D", "name": "delta data"},
    {"category_id": "E", "name": "epsilon data"},
    {"category_id": "F", "name": "zeta data"},
]


def test_choice_ids_are_sequential_and_cover_the_registry() -> None:
    registry = _registry(BASIC)
    choices = PromptChoiceRegistry.from_registry(registry)

    assert choices.choice_ids == ("1", "2", "3", "4", "5", "6")
    assert [choice.category_id for choice in choices.choices] == list(registry.ids)
    assert len({choice.choice_id for choice in choices.choices}) == len(choices.choices)
    assert all(
        choices.contains_category_id(category_id) for category_id in registry.ids
    )


def test_bidirectional_mapping_and_roundtrip() -> None:
    registry = _registry(BASIC)
    choices = PromptChoiceRegistry.from_registry(registry)

    assert choices.choice_id_of("A") == "1"
    assert choices.category_id_of("6") == "F"
    for category in registry.categories:
        choice_id = choices.choice_id_of(category.category_id)
        assert choices.category_id_of(choice_id) == category.category_id


def test_mapping_is_deterministic() -> None:
    first = PromptChoiceRegistry.from_registry(_registry(BASIC))
    second = PromptChoiceRegistry.from_registry(_registry(BASIC))

    assert first.choices == second.choices
    assert first.choice_ids == second.choice_ids


def test_unique_leaf_keeps_the_leaf_name() -> None:
    registry = _registry(BASIC)
    choices = PromptChoiceRegistry.from_registry(registry)

    assert choices.display_name_of("A") == "alpha data"


def test_duplicate_leaf_uses_shortest_unique_parent_suffix() -> None:
    registry = _registry(
        [
            {
                "category_id": "finance:业务.账户信息.基本信息",
                "name": "基本信息",
                "path": ["业务", "账户信息", "基本信息"],
            },
            {
                "category_id": "finance:业务.合约协议.基本信息",
                "name": "基本信息",
                "path": ["业务", "合约协议", "基本信息"],
            },
            {"category_id": "X", "name": "个人联系信息", "path": ["个人联系信息"]},
            {"category_id": "Y", "name": "个人财产信息", "path": ["个人财产信息"]},
            {"category_id": "Z", "name": "个人健康生理信息", "path": ["个人健康生理信息"]},
        ]
    )
    choices = PromptChoiceRegistry.from_registry(registry)

    assert choices.display_name_of("finance:业务.账户信息.基本信息") == "账户信息 / 基本信息"
    assert choices.display_name_of("finance:业务.合约协议.基本信息") == "合约协议 / 基本信息"
    assert choices.display_name_of("X") == "个人联系信息"


def test_two_parent_levels_needed_until_unique() -> None:
    """A/B/X and C/B/X still collide at 'B / X', so both need 3 levels."""
    registry = _registry(
        [
            {"category_id": "c1", "name": "X", "path": ["A", "B", "X"]},
            {"category_id": "c2", "name": "X", "path": ["A", "C", "X"]},
            {"category_id": "c3", "name": "X", "path": ["C", "B", "X"]},
            {"category_id": "c4", "name": "Y", "path": ["Y"]},
            {"category_id": "c5", "name": "Z", "path": ["Z"]},
        ]
    )
    choices = PromptChoiceRegistry.from_registry(registry)

    assert choices.display_name_of("c1") == "A / B / X"
    assert choices.display_name_of("c2") == "C / X"
    assert choices.display_name_of("c3") == "C / B / X"


def test_all_display_names_are_unique() -> None:
    registry = _registry(
        [
            {"category_id": "c1", "name": "基本信息", "path": ["业务", "账户信息", "基本信息"]},
            {"category_id": "c2", "name": "基本信息", "path": ["业务", "合约协议", "基本信息"]},
            {"category_id": "c3", "name": "基本信息", "path": ["经营管理", "营销服务", "基本信息"]},
            {"category_id": "c4", "name": "行为信息", "path": ["客户", "个人", "行为信息"]},
            {"category_id": "c5", "name": "行为信息", "path": ["客户", "单位", "行为信息"]},
        ]
    )
    choices = PromptChoiceRegistry.from_registry(registry)

    names = [choice.display_name for choice in choices.choices]
    assert len(set(names)) == len(names)
    assert choices.display_name_of("c3") == "营销服务 / 基本信息"


def test_empty_paths_duplicate_leaves_fail_explicitly() -> None:
    registry = _registry(
        [
            {"category_id": "finance:业务.账户信息.基本信息", "name": "基本信息"},
            {"category_id": "finance:业务.合约协议.基本信息", "name": "基本信息"},
            {"category_id": "X", "name": "Y"},
            {"category_id": "Z", "name": "W"},
            {"category_id": "U", "name": "V"},
        ]
    )
    with pytest.raises(PromptChoiceError, match="cannot build a unique display name"):
        PromptChoiceRegistry.from_registry(registry)


def test_identical_full_paths_fail_explicitly() -> None:
    registry = _registry(
        [
            {"category_id": "c1", "name": "X", "path": ["A", "X"]},
            {"category_id": "c2", "name": "X", "path": ["A", "X"]},
            {"category_id": "c3", "name": "Y"},
            {"category_id": "c4", "name": "Z"},
            {"category_id": "c5", "name": "W"},
        ]
    )
    with pytest.raises(PromptChoiceError, match="cannot build a unique display name"):
        PromptChoiceRegistry.from_registry(registry)


def test_from_registry_never_mutates_leaf_categories() -> None:
    registry = _registry(BASIC)
    snapshot = registry.categories
    PromptChoiceRegistry.from_registry(registry)

    assert registry.categories == snapshot
    assert registry.categories[0].name == "alpha data"
    assert registry.categories[0].category_id == "A"


def test_mapping_failures_raise_prompt_choice_error() -> None:
    choices = PromptChoiceRegistry.from_registry(_registry(BASIC))

    with pytest.raises(PromptChoiceError, match="no prompt choice"):
        choices.choice_id_of("Z")
    with pytest.raises(PromptChoiceError, match="not in the prompt catalog"):
        choices.category_id_of("42")
    with pytest.raises(PromptChoiceError, match="no prompt choice"):
        choices.display_name_of("42")


def test_stage1_encode_decode_roundtrip() -> None:
    choices = PromptChoiceRegistry.from_registry(_registry(BASIC))
    canonical = ("C", "A", "B", "D", "E")

    encoded = choices.encode_candidates(canonical)
    assert encoded == ("3", "1", "2", "4", "5")
    assert choices.decode_candidates(encoded) == canonical


def test_stage1_decode_rejects_wrong_shape() -> None:
    choices = PromptChoiceRegistry.from_registry(_registry(BASIC))

    with pytest.raises(PromptChoiceError, match="exactly 5"):
        choices.decode_candidates(("1", "2", "3"))
    with pytest.raises(PromptChoiceError, match="unique"):
        choices.decode_candidates(("1", "1", "2", "4", "5"))
    with pytest.raises(PromptChoiceError, match="not in the prompt catalog"):
        choices.decode_candidates(("1", "2", "3", "4", "42"))


def test_stage2_local_ids_encode_and_decode_positionally() -> None:
    candidates = ("C", "A", "B", "D", "E")

    assert encode_stage2_answer("C", candidates) == "1"
    assert encode_stage2_answer("E", candidates) == "5"
    assert decode_stage2_answer("1", candidates) == "C"
    assert decode_stage2_answer("5", candidates) == "E"


def test_stage2_decode_rejects_non_local_ids() -> None:
    candidates = ("C", "A", "B", "D", "E")

    with pytest.raises(PromptChoiceError, match="one of 1..5"):
        decode_stage2_answer("6", candidates)
    with pytest.raises(PromptChoiceError, match="one of 1..5"):
        decode_stage2_answer("A", candidates)
    with pytest.raises(PromptChoiceError, match="one of 1..5"):
        decode_stage2_answer("01", candidates)
    with pytest.raises(PromptChoiceError, match="one of the candidates"):
        encode_stage2_answer("F", candidates)
