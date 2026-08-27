"""Generator/validator tests for the tool-trajectory SFT pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from agent.task import GradingConfig, LeafRegistry, TaskConfig
from agent.task.assets import load_corpus_categories
from agent.training.rl.native_tools import CategoryToolEnvironment, parse_final_tool_answer
from agent.training.rl.sample import NATIVE_TOOL_TRAJECTORY_FORMAT, build_native_tool_prompt
from agent.training.sft.tool_trajectories import (
    TRAJECTORY_CLASSES,
    build_tool_trajectory_prompt,
    export_tool_trajectory_dataset,
    render_tool_call,
    select_trajectory_class,
    validate_tool_trajectory_dataset,
)


ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "tests" / "sft" / "fixtures" / "tool_registry.json"
CORPUS = ROOT / "tests" / "sft" / "fixtures" / "tool_corpus.json"
CANONICAL = ROOT / "tests" / "sft" / "fixtures" / "tool_trajectory_canonical.json"
GRADING = ROOT / "tests" / "sft" / "fixtures" / "grading.json"
DEMO_REGISTRY = ROOT / "cfg" / "task" / "leaf_registry.example.json"
DEMO_CORPUS = ROOT / "cfg" / "task" / "corpus.example.json"
FIELDS_4 = (
    "field_name",
    "table_name",
    "field_description",
    "table_description",
)
FIELDS_2 = ("field_name", "table_name")


def _registry() -> LeafRegistry:
    return LeafRegistry.from_path(REGISTRY)


def _corpus_map() -> dict[str, object]:
    return {
        category.category_id: category
        for category in load_corpus_categories(CORPUS)
    }


def _grading() -> GradingConfig:
    return GradingConfig.from_path(GRADING)


def _config(fields: tuple[str, ...]) -> TaskConfig:
    return TaskConfig.from_mapping(
        {"task_name": "synthetic_field_classification", "metadata_fields": list(fields)}
    )


def _export(
    tmp_path: Path,
    canonical: Path = CANONICAL,
    fields: tuple[str, ...] = FIELDS_4,
    *,
    dataset: str = "shougang",
    **kwargs: object,
):
    output = tmp_path / "release"
    report = export_tool_trajectory_dataset(
        canonical,
        output,
        _registry(),
        corpus=_corpus_map(),
        task_config=_config(fields),
        grading=_grading(),
        dataset=dataset,
        **kwargs,
    )
    return output, report


def _rows(output: Path, split: str = "train") -> list[dict]:
    return pq.read_table(output / f"{split}.parquet").to_pylist()


# --- schema and classes -----------------------------------------------------


def test_all_four_classes_present_in_every_split(tmp_path: Path) -> None:
    output, report = _export(tmp_path)
    assert report["label_gap_gate"]["status"] == "passed"
    for split in ("train", "val", "test"):
        counts = report["splits"][split]["trajectory_class_counts"]
        assert set(counts) == set(TRAJECTORY_CLASSES)
        assert all(count >= 1 for count in counts.values())


def test_direct_trajectory_has_no_tools(tmp_path: Path) -> None:
    output, _ = _export(tmp_path)
    for row in _rows(output):
        if row["trajectory_class"] != "direct":
            continue
        assert [m["role"] for m in row["messages"]] == ["system", "user", "assistant"]
        assert row["tool_calls"] == []
        assert row["trajectory_format"] == NATIVE_TOOL_TRAJECTORY_FORMAT
        assert row["stage"] == "tool_trajectory"
        assert row["dataset"] == "shougang"
        assert row["messages"][-1]["reasoning_content"].strip()
        assert row["think_truncated"] is False


@pytest.mark.parametrize(
    ("trajectory_class", "expected_tool_calls"),
    [("single_tool", 1), ("multi_tool", 3), ("no_result", 2)],
)
def test_tool_classes_follow_assistant_tool_alternation(
    tmp_path: Path, trajectory_class: str, expected_tool_calls: int
) -> None:
    output, _ = _export(tmp_path)
    seen = 0
    for row in _rows(output):
        if row["trajectory_class"] != trajectory_class:
            continue
        seen += 1
        roles = [m["role"] for m in row["messages"]]
        expected = ["system", "user"]
        for _ in range(expected_tool_calls):
            expected.extend(["assistant", "tool"])
        expected.append("assistant")
        assert roles == expected
        assert len(row["tool_calls"]) == expected_tool_calls
        assert [c["name"] for c in row["tool_calls"]][0] == "search_categories" or (
            trajectory_class == "no_result"
            and [c["name"] for c in row["tool_calls"]][0] == "browse_categories"
        )
    assert seen >= 1


def test_terminal_json_is_the_only_label_and_matches_ground_truth(
    tmp_path: Path,
) -> None:
    output, _ = _export(tmp_path)
    for split in ("train", "val", "test"):
        for row in _rows(output, split):
            terminal = row["messages"][-1]["content"]
            parsed = parse_final_tool_answer(
                terminal, registry=_registry(), grading=_grading()
            )
            assert parsed.category_id == row["ground_truth"]
            assert parsed.level == row["ground_truth_level"]
            # Only the terminal assistant message carries the answer object.
            for message in row["messages"][:-1]:
                if message["role"] == "assistant":
                    assert '"answer"' not in message["content"]


def test_render_tool_call_matches_qwen3_coder_format() -> None:
    call = render_tool_call(
        "get_category_examples",
        {"choice_ids": ["1", "2"], "limit": 2},
    )
    assert call == (
        "<tool_call><function=get_category_examples"
        "<parameter=choice_ids>[\"1\",\"2\"]</parameter>"
        "<parameter=limit>2</parameter></function></tool_call>"
    )
    search = render_tool_call(
        "search_categories", {"field_name": "alpha_field", "table_name": ""}
    )
    assert search == (
        "<tool_call><function=search_categories"
        "<parameter=field_name>alpha_field</parameter>"
        "<parameter=table_name></parameter></function></tool_call>"
    )


# --- leakage audit ----------------------------------------------------------


def _environment() -> CategoryToolEnvironment:
    return CategoryToolEnvironment(_registry(), _corpus_map())


def test_tool_results_are_byte_exact_environment_outputs(tmp_path: Path) -> None:
    output, _ = _export(tmp_path)
    env = _environment()
    for split in ("train", "val", "test"):
        for row in _rows(output, split):
            tool_messages = [m for m in row["messages"] if m["role"] == "tool"]
            for call_index, call in enumerate(row["tool_calls"]):
                name = call["name"]
                arguments = call["arguments"]
                if name == "search_categories":
                    if "scope" in arguments:
                        result = env.search_categories(
                            arguments["field_name"],
                            arguments["table_name"],
                            scope=arguments["scope"],
                        )
                    else:
                        result = env.search_categories(
                            arguments["field_name"], arguments["table_name"]
                        )
                elif name == "get_category_details":
                    result = env.get_category_details(arguments["choice_ids"])
                elif name == "get_category_examples":
                    result = env.get_category_examples(
                        arguments["choice_ids"], limit=arguments["limit"]
                    )
                elif name == "browse_categories":
                    result = env.browse_categories()
                else:  # pragma: no cover
                    raise AssertionError(f"unexpected call {name}")
                assert json.dumps(result, ensure_ascii=False) == tool_messages[
                    call_index
                ]["content"]


def test_no_leakage_ground_truth_absent_from_prompt(tmp_path: Path) -> None:
    output, _ = _export(tmp_path)
    for split in ("train", "val", "test"):
        for row in _rows(output, split):
            prompt_text = row["messages"][0]["content"] + row["messages"][1]["content"]
            assert row["ground_truth"] not in prompt_text


def _normalized_arguments(call: dict) -> dict:
    """Drop parquet struct-padding None keys from recorded tool arguments."""

    return {
        key: value
        for key, value in call.get("arguments", {}).items()
        if value is not None
    }


def test_no_result_never_surfaces_gt_choice_id(tmp_path: Path) -> None:
    output, _ = _export(tmp_path)
    env = _environment()
    for split in ("train", "val", "test"):
        for row in _rows(output, split):
            if row["trajectory_class"] != "no_result":
                continue
            gt_choice_id = env.choices.choice_id_of(row["ground_truth"])
            surfaced: set[str] = set()
            for message in row["messages"]:
                if message["role"] != "tool":
                    continue
                value = json.loads(message["content"])
                surfaced.update(_choice_ids_of_result(value))
            assert gt_choice_id not in surfaced
            # The terminal JSON still answers with the GT choice id.
            terminal = json.loads(row["messages"][-1]["content"])
            assert terminal["answer"] == gt_choice_id


def _choice_ids_of_result(value: object) -> set[str]:
    if not isinstance(value, dict):
        return set()
    ids: set[str] = set()
    for bucket in ("candidates", "categories", "leaves"):
        for entry in value.get(bucket, []) or []:
            choice_id = entry.get("choice_id") if isinstance(entry, dict) else None
            if isinstance(choice_id, str) and choice_id.strip():
                ids.add(choice_id)
    return ids


def test_tool_call_arguments_derive_from_metadata_and_prior_results(
    tmp_path: Path,
) -> None:
    output, _ = _export(tmp_path)
    for split in ("train", "val", "test"):
        for row in _rows(output, split):
            surfaced: set[str] = set()
            for call in row["tool_calls"]:
                name = call["name"]
                arguments = _normalized_arguments(call)
                if name == "search_categories":
                    assert arguments["field_name"] == row["metadata"].get(
                        "field_name", ""
                    )
                    assert arguments["table_name"] == row["metadata"].get(
                        "table_name", ""
                    )
                    if "scope" in arguments:
                        assert arguments["scope"] in {"Finance", "Hr"}
                        assert arguments["scope"] != _scope_of(
                            row["ground_truth"]
                        )
                elif name in ("get_category_details", "get_category_examples"):
                    assert set(arguments["choice_ids"]) <= surfaced
                for message in row["messages"]:
                    if message["role"] != "tool":
                        continue
                    value = json.loads(message["content"])
                    if "candidates" in value:
                        surfaced.update(_choice_ids_of_result(value))


def _scope_of(ground_truth: str) -> str:
    for category in _registry().categories:
        if category.category_id == ground_truth:
            return str(category.path[0])
    raise AssertionError(f"unexpected ground truth {ground_truth!r}")


def test_prompt_matches_rloo_seam_with_native_fields(tmp_path: Path) -> None:
    output, _ = _export(tmp_path, fields=FIELDS_4)
    registry = _registry()
    grading = _grading()
    config = _config(FIELDS_4)
    for row in _rows(output):
        native = build_native_tool_prompt(row["metadata"], grading, registry)
        mine = build_tool_trajectory_prompt(row["metadata"], grading, registry, config)
        assert native.system == mine.system
        assert native.user == mine.user


def test_metadata_fields_parameterized_two_field_config(tmp_path: Path) -> None:
    output, report = _export(tmp_path, fields=FIELDS_2)
    assert report["metadata_fields"] == list(FIELDS_2)
    validation = validate_tool_trajectory_dataset(
        output,
        _registry(),
        corpus=_corpus_map(),
        task_config=_config(FIELDS_2),
        grading=_grading(),
    )
    assert validation["valid"] is True
    for row in _rows(output):
        assert set(row["metadata"]) == set(FIELDS_2)


def test_export_is_deterministic(tmp_path: Path) -> None:
    first, _ = _export(tmp_path / "a")
    second, _ = _export(tmp_path / "b")
    for split in ("train", "val", "test"):
        assert (first / f"{split}.parquet").read_bytes() == (
            second / f"{split}.parquet"
        ).read_bytes()


# --- shougang-only singleton contract ---------------------------------------


def test_dataset_must_be_shougang(tmp_path: Path) -> None:
    for dataset in ("finance", "shougang+finance", ""):
        with pytest.raises(ValueError, match="exactly.*shougang"):
            _export(tmp_path / dataset.replace("+", "_") or "empty", dataset=dataset)


# --- think-generator seam ----------------------------------------------------


def test_mock_think_is_deterministic_and_gt_free(tmp_path: Path) -> None:
    first, report = _export(tmp_path / "a")
    second, _ = _export(tmp_path / "b")
    assert report["think"]["generator"] == "mock"
    assert report["think"]["max_tokens"] == 128
    assert report["think"]["over_limit_policy"] == "truncate"
    assert report["think"]["tokenizer"] == "conservative_approx"
    assert report["think_truncated"] == 0
    for split in ("train", "val", "test"):
        for left, right in zip(_rows(first, split), _rows(second, split)):
            assert left["messages"][-1]["reasoning_content"] == right[
                "messages"
            ][-1]["reasoning_content"]
            reasoning = left["messages"][-1]["reasoning_content"]
            assert reasoning.strip()
            assert left["ground_truth"] not in reasoning


def _variable_think_generator() -> object:
    import hashlib as _hashlib

    class VariableThink:
        name = "variable"

        def generate(self, *, source_id: str, metadata: dict, trajectory_class: str) -> str:
            size = 10 + (
                int(_hashlib.sha256(f"think:{source_id}".encode()).hexdigest(), 16) % 40
            )
            return "x" * size

    return VariableThink()


def test_think_limit_truncates_over_budget_text(tmp_path: Path) -> None:
    output, report = _export(
        tmp_path, max_think_tokens=5, think_generator=_variable_think_generator()
    )
    assert report["think_truncated"] > 0
    assert report["think_discarded"] == 0
    validation = validate_tool_trajectory_dataset(
        output,
        _registry(),
        corpus=_corpus_map(),
        task_config=_config(FIELDS_4),
        grading=_grading(),
        max_think_tokens=5,
    )
    assert validation["valid"] is True
    assert any(
        row["think_truncated"]
        for split in ("train", "val", "test")
        for row in _rows(output, split)
    )


def test_think_limit_discards_over_budget_rows(tmp_path: Path) -> None:
    output, report = _export(
        tmp_path,
        max_think_tokens=5,
        think_over_limit="discard",
        think_generator=_variable_think_generator(),
    )
    assert report["think_discarded"] > 0
    assert report["think_truncated"] == 0
    total = sum(
        report["splits"][split]["exported_records"] for split in ("train", "val", "test")
    )
    assert total < 24
    validation = validate_tool_trajectory_dataset(
        output,
        _registry(),
        corpus=_corpus_map(),
        task_config=_config(FIELDS_4),
        grading=_grading(),
        max_think_tokens=5,
    )
    assert validation["valid"] is True


def test_estimate_think_tokens_approximation_and_tokenizer() -> None:
    from agent.training.sft.tool_trajectories import estimate_think_tokens

    class StubTokenizer:
        def encode(self, text: str) -> list[int]:
            return [1] * len(text)

    assert estimate_think_tokens("") == 0
    assert estimate_think_tokens("abc") == 1  # ceil(3 / 4) ASCII
    assert estimate_think_tokens("数据") == 2  # one token per CJK char
    assert estimate_think_tokens("abc", StubTokenizer()) == 3


def test_mock_think_has_no_answer_mask_doppelganger(tmp_path: Path) -> None:
    # Server answer_mask patch localizes `"answer"\s*:\s*"..."` value spans
    # in assistant content with first-match-wins; think text must not contain
    # quoted-key lookalikes that would mislabel a span (see design doc §5c).
    output, _ = _export(tmp_path)
    for split in ("train", "val", "test"):
        for row in _rows(output, split):
            think = row["messages"][-1]["reasoning_content"]
            assert '"answer"' not in think
            assert '"level"' not in think


def test_validation_rejects_missing_reasoning_content(tmp_path: Path) -> None:
    output, _ = _export(tmp_path)
    rows = _rows(output)
    del rows[0]["messages"][-1]["reasoning_content"]
    _write_mutated(output, rows)
    validation = validate_tool_trajectory_dataset(
        output, _registry(), corpus=_corpus_map(), task_config=_config(FIELDS_4),
        grading=_grading(),
    )
    assert validation["valid"] is False
    assert any(
        "reasoning_content" in error
        for error in validation["splits"]["train"]["errors"]
    )


def test_validation_rejects_gt_in_reasoning_content(tmp_path: Path) -> None:
    output, _ = _export(tmp_path)
    rows = _rows(output)
    rows[0]["messages"][-1]["reasoning_content"] = (
        "the answer is " + rows[0]["ground_truth"]
    )
    _write_mutated(output, rows)
    validation = validate_tool_trajectory_dataset(
        output, _registry(), corpus=_corpus_map(), task_config=_config(FIELDS_4),
        grading=_grading(),
    )
    assert validation["valid"] is False
    assert any(
        "reasoning_content must not expose" in error
        for error in validation["splits"]["train"]["errors"]
    )


def test_validation_rejects_over_limit_reasoning_content(tmp_path: Path) -> None:
    output, _ = _export(tmp_path)
    rows = _rows(output)
    rows[0]["messages"][-1]["reasoning_content"] = "长" * 5000
    _write_mutated(output, rows)
    validation = validate_tool_trajectory_dataset(
        output, _registry(), corpus=_corpus_map(), task_config=_config(FIELDS_4),
        grading=_grading(),
    )
    assert validation["valid"] is False
    assert any(
        "exceeds max_think_tokens" in error
        for error in validation["splits"]["train"]["errors"]
    )


# --- collect / file think source seam ---------------------------------------


def _collect(tmp_path: Path, **kwargs: object):
    from agent.training.sft.tool_trajectories import collect_tool_trajectory_contexts

    collect_dir = tmp_path / "collect"
    report = collect_tool_trajectory_contexts(
        CANONICAL,
        collect_dir,
        _registry(),
        corpus=_corpus_map(),
        task_config=_config(FIELDS_4),
        grading=_grading(),
        **kwargs,
    )
    return collect_dir, report


def _fill_shards(collect_dir: Path, *, with_gt: bool = False, mutate: dict | None = None) -> None:
    for shard in sorted(collect_dir.glob("*.jsonl")):
        entries = [json.loads(line) for line in shard.read_text(encoding="utf-8").splitlines()]
        for entry in entries:
            entry["think"] = (
                f"reasoning for {entry['sample_id']}"
                + (f" target {entry['ground_truth']}" if with_gt else "")
            )
            if mutate is not None and entry["sample_id"] == mutate.get("sample_id"):
                entry["think"] = mutate["think"]
        shard.write_text(
            "\n".join(json.dumps(entry, ensure_ascii=False) for entry in entries) + "\n",
            encoding="utf-8",
        )


def _export_file(tmp_path: Path, collect_dir: Path, **kwargs: object):
    from agent.training.sft.tool_trajectories import FileThinkGenerator

    output = tmp_path / "release"
    report = export_tool_trajectory_dataset(
        CANONICAL,
        output,
        _registry(),
        corpus=_corpus_map(),
        task_config=_config(FIELDS_4),
        grading=_grading(),
        think_generator=FileThinkGenerator(collect_dir),
        **kwargs,
    )
    return output, report


def test_collect_shards_cover_all_records_and_carry_full_context(tmp_path: Path) -> None:
    collect_dir, report = _collect(tmp_path, shard_size=8)
    assert report["format"] == "verl_tool_trajectory_think_collect_v1"
    assert report["label_aware"] is True
    assert report["total_samples"] == 24
    assert report["splits"] == {"train": 8, "val": 8, "test": 8}
    assert len(report["shards"]) == 3  # one shard per split at shard_size=8
    rows = [
        json.loads(line)
        for shard in sorted(collect_dir.glob("*.jsonl"))
        for line in shard.read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 24
    assert all(row["think"] == "" for row in rows)
    for row in rows:
        assert row["sample_id"].startswith(("train-", "val-", "test-"))
        assert row["trajectory_class"] in TRAJECTORY_CLASSES
        assert list(row["prompt"]) == ["system", "user"]
        assert row["ground_truth"] in _registry().ids
        assert row["ground_truth_level"] in _grading().levels
        assert set(row["terminal_json"]) == {"answer", "level"}
        assert isinstance(row["tool_calls"], list)
        if row["trajectory_class"] == "direct":
            assert row["tool_calls"] == []
            assert row["tool_results"] == []
        else:
            assert len(row["tool_calls"]) == len(row["tool_results"]) >= 1


def test_file_think_assemble_roundtrip(tmp_path: Path) -> None:
    collect_dir, _ = _collect(tmp_path)
    _fill_shards(collect_dir)
    output, report = _export_file(tmp_path, collect_dir)
    assert report["think"]["generator"] == "file"
    assert report["think_truncated"] == 0
    validation = validate_tool_trajectory_dataset(
        output, _registry(), corpus=_corpus_map(), task_config=_config(FIELDS_4),
        grading=_grading(),
    )
    assert validation["valid"] is True
    for split in ("train", "val", "test"):
        for row in _rows(output, split):
            assert row["think_source"] == "file"
            assert row["messages"][-1]["reasoning_content"] == (
                f"reasoning for {row['source_id']}"
            )


def test_file_think_missing_sample_fails_fast(tmp_path: Path) -> None:
    collect_dir, _ = _collect(tmp_path)
    _fill_shards(collect_dir)
    # drop one sample from every shard
    dropped = "train-00"
    for shard in sorted(collect_dir.glob("*.jsonl")):
        entries = [
            json.loads(line)
            for line in shard.read_text(encoding="utf-8").splitlines()
            if json.loads(line)["sample_id"] != dropped
        ]
        shard.write_text(
            "\n".join(json.dumps(entry, ensure_ascii=False) for entry in entries) + "\n",
            encoding="utf-8",
        )
    with pytest.raises(ValueError, match=dropped):
        _export_file(tmp_path, collect_dir)


def test_file_think_allows_bare_ground_truth_id(tmp_path: Path) -> None:
    # Label-aware distillation: the file think may mention the bare GT id.
    collect_dir, _ = _collect(tmp_path)
    _fill_shards(collect_dir, with_gt=True)
    output, _ = _export_file(tmp_path, collect_dir)
    validation = validate_tool_trajectory_dataset(
        output, _registry(), corpus=_corpus_map(), task_config=_config(FIELDS_4),
        grading=_grading(),
    )
    assert validation["valid"] is True
    assert any(
        f"target {row['ground_truth']}" in row["messages"][-1]["reasoning_content"]
        for split in ("train", "val", "test")
        for row in _rows(output, split)
    )


def test_file_think_rejects_terminal_json_object(tmp_path: Path) -> None:
    collect_dir, _ = _collect(tmp_path)
    _fill_shards(
        collect_dir,
        mutate={"sample_id": "train-00", "think": '{"answer": "1", "level": "L1"}'},
    )
    output, _ = _export_file(tmp_path, collect_dir)
    validation = validate_tool_trajectory_dataset(
        output, _registry(), corpus=_corpus_map(), task_config=_config(FIELDS_4),
        grading=_grading(),
    )
    assert validation["valid"] is False
    assert any(
        "terminal JSON object" in error
        for error in validation["splits"]["train"]["errors"]
    )


def test_think_rejects_unbalanced_brackets(tmp_path: Path) -> None:
    collect_dir, _ = _collect(tmp_path)
    _fill_shards(collect_dir, mutate={"sample_id": "train-00", "think": "unclosed {"})
    output, _ = _export_file(tmp_path, collect_dir)
    validation = validate_tool_trajectory_dataset(
        output, _registry(), corpus=_corpus_map(), task_config=_config(FIELDS_4),
        grading=_grading(),
    )
    assert validation["valid"] is False
    assert any(
        "not closed" in error
        for error in validation["splits"]["train"]["errors"]
    )


def test_collect_context_matches_assemble_messages_byte_for_byte(
    tmp_path: Path,
) -> None:
    # Guard against future divergence between the collect path and the export
    # path: both derive from _build_trajectory_context, so prompts, tool
    # calls, tool results, and the terminal JSON must match byte-for-byte.
    collect_dir, _ = _collect(tmp_path)
    shard_entries = {
        json.loads(line)["sample_id"]: json.loads(line)
        for shard in sorted(collect_dir.glob("*.jsonl"))
        for line in shard.read_text(encoding="utf-8").splitlines()
    }
    _fill_shards(collect_dir)
    output, _ = _export_file(tmp_path, collect_dir)
    for split in ("train", "val", "test"):
        for row in _rows(output, split):
            entry = shard_entries[row["source_id"]]
            assert entry["prompt"]["system"] == row["messages"][0]["content"]
            assert entry["prompt"]["user"] == row["messages"][1]["content"]
            normalized = [
                {
                    "name": call["name"],
                    "arguments": {
                        key: value
                        for key, value in call["arguments"].items()
                        if value is not None
                    },
                }
                for call in row["tool_calls"]
            ]
            assert entry["tool_calls"] == normalized
            tool_results = [
                message["content"]
                for message in row["messages"]
                if message["role"] == "tool"
            ]
            assert entry["tool_results"] == tool_results
            assert entry["terminal_json"] == json.loads(
                row["messages"][-1]["content"]
            )
            assert entry["ground_truth"] == row["ground_truth"]
            assert entry["ground_truth_level"] == row["ground_truth_level"]
            assert entry["trajectory_class"] == row["trajectory_class"]


def test_truncate_backs_off_to_closed_prefix(tmp_path: Path) -> None:
    class UnclosedThink:
        name = "unclosed"

        def generate(self, *, source_id: str, metadata: dict, trajectory_class: str) -> str:
            return "a" * 30 + "{"

    output, report = _export(tmp_path, max_think_tokens=5, think_generator=UnclosedThink())
    assert report["think_truncated"] > 0
    validation = validate_tool_trajectory_dataset(
        output, _registry(), corpus=_corpus_map(), task_config=_config(FIELDS_4),
        grading=_grading(), max_think_tokens=5,
    )
    assert validation["valid"] is True
    for split in ("train", "val", "test"):
        for row in _rows(output, split):
            think = row["messages"][-1]["reasoning_content"]
            assert "{" not in think
            assert think.endswith("a")


def test_truncate_empty_closed_prefix_fails_fast(tmp_path: Path) -> None:
    class BracketsOnlyThink:
        name = "brackets"

        def generate(self, *, source_id: str, metadata: dict, trajectory_class: str) -> str:
            return "{" * 30

    with pytest.raises(ValueError, match="empty reasoning_content"):
        _export(tmp_path, max_think_tokens=5, think_generator=BracketsOnlyThink())


# --- gates and fail-fast ----------------------------------------------------


def _record(record_id: str, category: str, level: str, split: str) -> dict:
    name = category.rsplit(":", 1)[1].capitalize()
    return {
        "schema_version": 2,
        "id": record_id,
        "resolution_status": "resolved",
        "metadata": {
            "field_name": f"{name.lower()}_field",
            "table_name": f"tbl_{name.lower()}",
            "field_description": f"Source column {name.lower()}_field.",
            "table_description": f"Synthetic table for {name.lower()}.",
        },
        "classification": {"group": name, "category": name},
        "data_level": level,
        "target": {
            "leaf_level": "category",
            "leaf_name": name,
            "category_id": category,
            "category_path": ["Finance", name],
        },
        "split": split,
        "split_exclusion_reason": None,
    }


def test_label_gap_gate_fails_without_waiver(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical.json"
    rows = [
        _record("tr-0", "demo:alpha", "L1", "train"),
        _record("tr-1", "demo:bravo", "L2", "train"),
        _record("tr-2", "demo:charlie", "L3", "train"),
        _record("va-0", "demo:alpha", "L1", "val"),
        # delta L4 appears only in test: category AND level gap.
        _record("te-0", "demo:delta", "L4", "test"),
    ]
    canonical.write_text(json.dumps(rows), encoding="utf-8")
    with pytest.raises(ValueError, match="label-gap gate failed"):
        _export(tmp_path / "x", canonical=canonical)
    report = export_tool_trajectory_dataset(
        canonical,
        tmp_path / "y",
        _registry(),
        corpus=_corpus_map(),
        task_config=_config(FIELDS_4),
        grading=_grading(),
        allow_label_gaps=("demo:delta", "L4"),
    )
    assert report["label_gap_gate"]["status"] == "waived"


def test_single_group_registry_no_result_fails_fast(tmp_path: Path) -> None:
    no_result_id = next(
        candidate
        for candidate in (f"sg-{index}" for index in range(20))
        if select_trajectory_class(candidate) == "no_result"
    )
    name = no_result_id.replace("-", "_")
    record = {
        "schema_version": 2,
        "id": no_result_id,
        "resolution_status": "resolved",
        "metadata": {"field_name": "alpha_field", "table_name": "tbl_alpha"},
        "classification": {"group": "Alpha", "category": "Alpha"},
        "data_level": "L1",
        "target": {
            "leaf_level": "category",
            "leaf_name": "Alpha",
            "category_id": "demo:alpha",
            "category_path": ["Synthetic", "Alpha"],
        },
        "split": "train",
        "split_exclusion_reason": None,
    }
    canonical = tmp_path / "canonical.json"
    canonical.write_text(json.dumps([record]), encoding="utf-8")
    demo_corpus = {
        category.category_id: category
        for category in load_corpus_categories(DEMO_CORPUS)
    }
    with pytest.raises(ValueError, match="at least two level-1 groups"):
        export_tool_trajectory_dataset(
            canonical,
            tmp_path / "x",
            LeafRegistry.from_path(DEMO_REGISTRY),
            corpus=demo_corpus,
            task_config=_config(FIELDS_2),
            grading=_grading(),
        )


def test_field_name_required_for_tool_classes(tmp_path: Path) -> None:
    record = _record("tr-0", "demo:alpha", "L1", "train")
    record["metadata"]["field_name"] = ""
    canonical = tmp_path / "canonical.json"
    canonical.write_text(json.dumps([record]), encoding="utf-8")
    if select_trajectory_class("tr-0") == "direct":
        pytest.skip("this id maps to direct; tool classes are not exercised")
    with pytest.raises(ValueError, match="field_name"):
        _export(tmp_path / "x", canonical=canonical)


# --- validator robustness ---------------------------------------------------


def _mutate_row(row: dict, **changes: object) -> dict:
    mutated = json.loads(json.dumps(row))
    mutated.update(changes)
    return mutated


def test_validation_rejects_mutated_terminal_json(tmp_path: Path) -> None:
    output, _ = _export(tmp_path)
    rows = _rows(output)
    rows[0]["messages"][-1]["content"] = '{"answer": "9", "level": "L1"}'
    _write_mutated(output, rows)
    validation = validate_tool_trajectory_dataset(
        output, _registry(), corpus=_corpus_map(), task_config=_config(FIELDS_4),
        grading=_grading(),
    )
    assert validation["valid"] is False
    assert any("terminal" in error for error in validation["splits"]["train"]["errors"])


def test_validation_rejects_fake_tool_result(tmp_path: Path) -> None:
    output, _ = _export(tmp_path)
    rows = _rows(output)
    tool_rows = [row for row in rows if row["tool_calls"]]
    tool_rows[0]["messages"][3]["content"] = '{"candidates": [{"choice_id": "9"}]}'
    _write_mutated(output, rows)
    validation = validate_tool_trajectory_dataset(
        output, _registry(), corpus=_corpus_map(), task_config=_config(FIELDS_4),
        grading=_grading(),
    )
    assert validation["valid"] is False
    assert any(
        "byte-exact environment result" in error
        for error in validation["splits"]["train"]["errors"]
    )


def test_validation_rejects_serialized_chat_tokens(tmp_path: Path) -> None:
    output, _ = _export(tmp_path)
    rows = _rows(output)
    rows[0]["messages"][1]["content"] += "<|im_start|>user"
    _write_mutated(output, rows)
    validation = validate_tool_trajectory_dataset(
        output, _registry(), corpus=_corpus_map(), task_config=_config(FIELDS_4),
        grading=_grading(),
    )
    assert validation["valid"] is False
    assert any(
        "chat-template tokens" in error
        for error in validation["splits"]["train"]["errors"]
    )


def test_validation_catches_duplicate_source_id_and_cross_split(tmp_path: Path) -> None:
    output, _ = _export(tmp_path)
    validation = validate_tool_trajectory_dataset(
        output, _registry(), corpus=_corpus_map(), task_config=_config(FIELDS_4),
        grading=_grading(),
    )
    assert validation["valid"] is True
    assert validation["cross_split_errors"] == []
    rows = _rows(output)
    rows.append(_mutate_row(rows[0], source_id=rows[0]["source_id"]))
    _write_mutated(output, rows)
    validation = validate_tool_trajectory_dataset(
        output, _registry(), corpus=_corpus_map(), task_config=_config(FIELDS_4),
        grading=_grading(),
    )
    assert validation["valid"] is False
    assert any("duplicate source_id" in error for error in validation["splits"]["train"]["errors"])


def _write_mutated(output: Path, rows: list[dict]) -> None:
    import pyarrow as pa

    pq.write_table(pa.Table.from_pylist(rows), output / "train.parquet")
