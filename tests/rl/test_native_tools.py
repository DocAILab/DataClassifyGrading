from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.task import GradingConfig, LeafRegistry
from agent.task.assets import load_corpus_categories
from agent.training.rl.native_tools import (
    CategoryToolEnvironment,
    exact_tool_reward,
    parse_final_tool_answer,
)


ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "cfg" / "task" / "leaf_registry.example.json"
CORPUS = ROOT / "cfg" / "task" / "corpus.example.json"


def _environment() -> CategoryToolEnvironment:
    registry = LeafRegistry.from_path(REGISTRY)
    corpus = {item.category_id: item for item in load_corpus_categories(CORPUS)}
    return CategoryToolEnvironment(registry, corpus)


def test_search_is_deterministic_fixed_top5_and_hides_canonical_ids() -> None:
    environment = _environment()
    first = environment.search_categories("Alpha", "demo_table")
    second = environment.search_categories("Alpha", "demo_table")

    assert first == second
    assert len(first["candidates"]) == 5
    assert first["candidates"][0]["choice_id"] == "1"
    assert all(set(item) == {"choice_id", "name", "summary"} for item in first["candidates"])
    assert "demo:" not in json.dumps(first)

    with pytest.raises(ValueError, match="fixed at 5"):
        environment.search_categories("Alpha", "demo_table", top_k=4)


def test_helpers_are_batched_bounded_and_use_only_opaque_ids() -> None:
    environment = _environment()
    details = environment.get_category_details(["2", "1"])
    examples = environment.get_category_examples(["1"], limit=1)

    assert [item["choice_id"] for item in details["categories"]] == ["2", "1"]
    assert examples["categories"][0]["examples"] == ["fabricated-alpha-example"]
    assert "demo:" not in json.dumps(details)
    assert "demo:" not in json.dumps(examples)

    with pytest.raises(ValueError, match="unknown opaque"):
        environment.get_category_details(["999"])
    with pytest.raises(ValueError, match="unique"):
        environment.get_category_examples(["1", "1"])
    with pytest.raises(ValueError, match="between 1 and 5"):
        environment.get_category_examples(["1"], limit=6)


def test_corpus_must_exactly_cover_registry() -> None:
    registry = LeafRegistry.from_path(REGISTRY)
    corpus = {item.category_id: item for item in load_corpus_categories(CORPUS)}
    corpus.pop("demo:alpha")
    with pytest.raises(ValueError, match="exactly cover"):
        CategoryToolEnvironment(registry, corpus)


def test_final_answer_is_strict_global_choice_and_joint_exact_match() -> None:
    registry = LeafRegistry.from_path(REGISTRY)
    grading = GradingConfig(("L1", "L2", "L3", "L4"))
    answer = parse_final_tool_answer(
        '{"answer":"1","level":"L2"}', registry=registry, grading=grading
    )
    assert answer.choice_id == "1"
    assert answer.category_id == "demo:alpha"
    assert exact_tool_reward(
        '{"answer":"1","level":"L2"}',
        ground_truth="demo:alpha",
        ground_truth_level="L2",
        registry=registry,
        grading=grading,
    ) == 1.0
    assert exact_tool_reward(
        '{"answer":"1","level":"L1"}',
        ground_truth="demo:alpha",
        ground_truth_level="L2",
        registry=registry,
        grading=grading,
    ) == 0.0


@pytest.mark.parametrize(
    "text",
    [
        'result: {"answer":"1","level":"L2"}',
        '```json\n{"answer":"1","level":"L2"}\n```',
        '{"answer":"demo:alpha","level":"L2"}',
        '{"answer":"1","level":"L2","extra":true}',
        '{"answer":"1","answer":"2","level":"L2"}',
        '{"answer":"1","level":"UNKNOWN"}',
        '{"candidates":["1","2","3","4","5"]}',
    ],
)
def test_final_answer_rejects_old_manual_or_nonexact_shapes(text: str) -> None:
    registry = LeafRegistry.from_path(REGISTRY)
    grading = GradingConfig(("L1", "L2", "L3", "L4"))
    with pytest.raises(ValueError):
        parse_final_tool_answer(text, registry=registry, grading=grading)


def test_scope_restricts_search_and_browse_lists_groups() -> None:
    environment = _environment()
    groups = environment.browse_categories()
    assert groups["groups"] and all(
        set(item) == {"scope_key", "leaf_count"} for item in groups["groups"]
    )
    key = groups["groups"][0]["scope_key"]
    subtree = environment.browse_categories(key)
    assert subtree["scope_key"] == key
    assert len(subtree["leaves"]) == subtree["leaf_count"]

    scoped = environment.search_categories("Alpha", "demo_table", scope=key)
    known_ids = {leaf["choice_id"] for leaf in subtree["leaves"]}
    assert all(item["choice_id"] in known_ids for item in scoped["candidates"])

    with pytest.raises(ValueError, match="unknown scope"):
        environment.search_categories("Alpha", "demo_table", scope="nope")


def test_v2b_scoring_is_deterministic_and_downweights_ubiquitous_tokens() -> None:
    environment = _environment()
    first = environment.search_categories("Alpha", "demo_table")
    second = environment.search_categories("Alpha", "demo_table")
    assert first == second


def test_render_catalog_index_and_four_field_prompt() -> None:
    from agent.task import GradingConfig
    from agent.training.rl.sample import (
        NATIVE_TOOL_TRAJECTORY_FORMAT,
        PROMPT_METADATA_FIELDS,
        build_native_tool_prompt,
        render_catalog_index,
    )

    registry = LeafRegistry.from_path(REGISTRY)
    index = render_catalog_index(registry)
    lines = index.splitlines()
    assert len(lines) == len(registry.ids)
    assert all("|" in line for line in lines)

    grading = GradingConfig(("L1", "L2", "L3", "L4"))
    base = {
        "field_name": "EQ_CODE",
        "table_name": "SRL_BASIC_TEST",
        "field_description": "",
        "table_description": "",
    }
    prompt = build_native_tool_prompt(base, grading, registry)
    assert "Catalog (choice_id|name):" in prompt.system
    assert '"field_name":"EQ_CODE"' in prompt.user

    # Parquet round-trips reorder mapping keys alphabetically; the rendered
    # user content must stay byte-identical regardless of input dict order.
    shuffled = {
        "table_description": "",
        "field_description": "",
        "table_name": "SRL_BASIC_TEST",
        "field_name": "EQ_CODE",
    }
    assert build_native_tool_prompt(shuffled, grading, registry).user == prompt.user

    with pytest.raises(ValueError, match="native tool prompt metadata"):
        build_native_tool_prompt(
            {"field_name": "a", "table_name": "b"}, grading, registry
        )
