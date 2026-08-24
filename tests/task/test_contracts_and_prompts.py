import json

import pytest

from agent.task import (
    GradedTaskContext,
    GradingConfig,
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
        {"category_id": "demo:alpha", "name": "Alpha", "description": "fixture alpha"},
        {"category_id": "demo:bravo", "name": "Bravo", "description": "fixture bravo"},
        {"category_id": "demo:charlie", "name": "Charlie", "description": "fixture charlie"},
        {"category_id": "demo:delta", "name": "Delta", "description": "fixture delta"},
        {"category_id": "demo:echo", "name": "Echo", "description": "fixture echo"},
        {"category_id": "demo:foxtrot", "name": "Foxtrot", "description": "fixture foxtrot"},
    ]
}
CONFIG = {"metadata_fields": ["title", "summary"]}
METADATA = {"title": "fabricated title", "summary": "fabricated summary", "hidden": "omit"}


def test_prompts_use_choice_protocol_without_canonical_ids() -> None:
    registry = LeafRegistry.from_mapping(REGISTRY)
    config = TaskConfig.from_mapping(CONFIG)
    candidates = ["demo:charlie", "demo:alpha", "demo:bravo", "demo:delta", "demo:echo"]

    stage1 = build_stage1_prompt(METADATA, registry, config)
    stage2 = build_stage2_prompt(METADATA, candidates, registry, config)

    assert '"candidates"' in stage1.system
    assert '"answer"' in stage2.system
    assert "demo:" not in stage1.user
    assert '"category_id"' not in stage1.user + stage2.user
    assert "fixture alpha" in stage1.user
    catalog = json.loads(
        stage1.user.split("catalog:\n", 1)[1].split("\nField metadata:", 1)[0]
    )
    assert catalog[0] == ["1", "Alpha", "fixture alpha"]
    assert "hidden" not in stage1.user
    bundle = json.loads(stage2.user.split("Candidate bundle:\n", 1)[1].split("\nField metadata:", 1)[0])
    assert [entry["id"] for entry in bundle] == ["1", "2", "3", "4", "5"]


def test_duplicate_names_use_synthetic_parent_suffixes() -> None:
    registry = LeafRegistry.from_mapping(
        {
            "categories": [
                {"category_id": "demo:a.common", "name": "Common", "path": ["Branch A", "Common"]},
                {"category_id": "demo:b.common", "name": "Common", "path": ["Branch B", "Common"]},
                {"category_id": "demo:c", "name": "Charlie"},
                {"category_id": "demo:d", "name": "Delta"},
                {"category_id": "demo:e", "name": "Echo"},
            ]
        }
    )
    choices = PromptChoiceRegistry.from_registry(registry)
    assert choices.display_name_of("demo:a.common") == "Branch A / Common"
    assert choices.display_name_of("demo:b.common") == "Branch B / Common"


def test_answer_builders_roundtrip_canonical_ids() -> None:
    registry = LeafRegistry.from_mapping(REGISTRY)
    choices = PromptChoiceRegistry.from_registry(registry)
    candidates = ["demo:charlie", "demo:alpha", "demo:bravo", "demo:delta", "demo:echo"]

    assert stage1_answer(candidates, choices=choices) == '{"candidates":["3","1","2","4","5"]}'
    assert choices.decode_candidates(("3", "1", "2", "4", "5")) == tuple(candidates)
    assert stage2_answer("demo:charlie", candidates) == '{"answer":"1"}'


def test_contracts_reject_invalid_shapes() -> None:
    with pytest.raises(ValueError, match="at least 5"):
        LeafRegistry.from_mapping(["a", "b", "c", "d"])
    with pytest.raises(ValueError, match="must be strings"):
        TaskConfig.from_mapping({"metadata_fields": [1]})


def test_grading_config_is_strict_and_rejects_unknown_fields() -> None:
    config = GradingConfig.from_mapping(
        {"levels": ["L1", "L2"], "descriptions": ["low", "high"]}
    )
    assert config.levels == ("L1", "L2")
    with pytest.raises(ValueError, match="levels.*strings"):
        GradingConfig.from_mapping({"levels": ["L1", 2]})
    with pytest.raises(ValueError, match="descriptions.*strings"):
        GradingConfig.from_mapping(
            {"levels": ["L1"], "descriptions": ["ok", 3]}
        )
    with pytest.raises(ValueError, match="unknown"):
        GradingConfig.from_mapping({"levels": ["L1"], "unexpected": True})


def test_graded_context_requires_both_heads_or_neither() -> None:
    grading = GradingConfig(levels=("L1", "L2"))
    assert GradedTaskContext().enabled is False
    assert GradedTaskContext(grading, "L1").enabled is True
    with pytest.raises(ValueError, match="together"):
        GradedTaskContext(grading=grading)
    with pytest.raises(ValueError, match="together"):
        GradedTaskContext(expected_level="L1")
    with pytest.raises(ValueError, match="levels"):
        GradedTaskContext(grading, "L3")
