import pytest

from agent.task import (
    LeafRegistry,
    TaskConfig,
    build_stage1_prompt,
    build_stage2_prompt,
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
    stage1 = build_stage1_prompt(METADATA, registry, config)
    stage2 = build_stage2_prompt(
        METADATA,
        ["C", "A", "B", "D", "E"],
        registry,
        config,
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
