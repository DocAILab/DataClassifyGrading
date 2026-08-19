import pytest

from agent.task import (
    LeafRegistry,
    PromptChoiceRegistry,
    TaskConfig,
    build_stage1_prompt,
    build_stage2_prompt,
    stage1_answer,
    stage2_answer,
)


REGISTRY = {
    "categories": [
        {"category_id": "A", "description": "alpha data"},
        {"category_id": "B", "description": "beta data"},
        {"category_id": "C", "description": "gamma data"},
        {"category_id": "D", "description": "delta data"},
        {"category_id": "E", "description": "epsilon data"},
        {"category_id": "F", "description": "zeta data"},
    ]
}
CONFIG = {"metadata_fields": ["field_name", "field_description"]}
METADATA = {
    "field_name": "email",
    "field_description": "contact address",
    "table_name": "secret_table",
}


def test_prompts_have_strict_stage_contracts_and_no_chatml_tokens() -> None:
    registry = LeafRegistry.from_mapping(REGISTRY)
    config = TaskConfig.from_mapping(CONFIG)
    choices = PromptChoiceRegistry.from_registry(registry)
    stage1 = build_stage1_prompt(METADATA, registry, config, choices=choices)
    stage2 = build_stage2_prompt(
        METADATA,
        ["C", "A", "B", "D", "E"],
        registry,
        config,
        choices=choices,
    )

    assert '"candidates"' in stage1.system and "exactly 5" in stage1.system
    assert '"answer"' in stage2.system and "one of the five" in stage2.system
    prompt_text = stage1.system + stage1.user + stage2.system + stage2.user
    assert all(
        token not in prompt_text
        for token in ("<|im_start|>", "<|im_end|>", "ChatML")
    )
    assert "secret_table" not in stage1.user
    assert '"field_name"' in stage1.user


def test_stage1_prompt_uses_choice_ids_and_never_canonical_ids() -> None:
    registry = LeafRegistry.from_mapping(REGISTRY)
    config = TaskConfig.from_mapping(CONFIG)
    stage1 = build_stage1_prompt(METADATA, registry, config)

    # compact [choice_id, display_name] pairs; no category_id anywhere
    assert '["1", "A"]' in stage1.user
    assert '"category_id"' not in stage1.user
    assert "finance:" not in stage1.user
    # descriptions stay out of Stage 1
    assert "alpha data" not in stage1.user


def test_stage1_prompt_disambiguates_duplicate_leaf_names() -> None:
    registry = LeafRegistry.from_mapping(
        {
            "categories": [
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
        }
    )
    config = TaskConfig.from_mapping(CONFIG)
    stage1 = build_stage1_prompt(METADATA, registry, config)

    assert '"账户信息 / 基本信息"' in stage1.user
    assert '"合约协议 / 基本信息"' in stage1.user
    assert "finance:业务.账户信息.基本信息" not in stage1.user
    assert '"category_id"' not in stage1.user


def test_stage2_prompt_uses_local_bundle_ids() -> None:
    registry = LeafRegistry.from_mapping(REGISTRY)
    config = TaskConfig.from_mapping(CONFIG)
    stage2 = build_stage2_prompt(
        METADATA,
        ["C", "A", "B", "D", "E"],
        registry,
        config,
    )
    user = stage2.user

    # local ids follow candidate order, not canonical ids
    assert '"id":"1"' in user and '"name":"C"' in user
    assert '"id":"5"' in user and '"name":"E"' in user
    assert '"category_id"' not in user
    assert '"answer"' in stage2.system


def test_answer_builders_use_choice_ids_and_decode_restores_canonical() -> None:
    registry = LeafRegistry.from_mapping(REGISTRY)
    choices = PromptChoiceRegistry.from_registry(registry)
    canonical = ["C", "A", "B", "D", "E"]

    stage1 = stage1_answer(canonical, choices=choices)
    assert stage1 == '{"candidates":["3","1","2","4","5"]}'
    assert choices.decode_candidates(["3", "1", "2", "4", "5"]) == tuple(canonical)

    stage2 = stage2_answer("C", canonical)
    assert stage2 == '{"answer":"1"}'


def test_invalid_registry_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least 5"):
        LeafRegistry.from_mapping({"categories": [{"category_id": "A"}] * 4})
    with pytest.raises(ValueError, match="unique"):
        LeafRegistry.from_mapping(
            {"categories": [{"category_id": "A", "description": "x"}] * 5}
        )
    with pytest.raises(ValueError, match="non-empty"):
        LeafRegistry.from_mapping([" ", "  ", "   ", "    ", "     "])


def test_task_config_rejects_non_string_metadata_fields() -> None:
    with pytest.raises(ValueError, match="must be strings"):
        TaskConfig.from_mapping({"metadata_fields": [1]})
