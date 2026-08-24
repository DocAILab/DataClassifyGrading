"""Joint classification+grading Stage 2 head (GradingConfig) end-to-end."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.evaluation import evaluate_stage2_choices
from agent.task import GradingConfig, LeafRegistry, TaskConfig
from agent.task.assets import load_corpus_categories
from agent.task.parser import parse_stage2_output
from agent.task.prompts import build_stage2_prompt, stage2_answer
from agent.training.rl.reward import reward_stage2_choices
from agent.training.sft import export_sft_dataset, validate_sft_dataset


ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "cfg" / "task" / "leaf_registry.example.json"
CORPUS = ROOT / "cfg" / "task" / "corpus.example.json"
GRADING = ROOT / "cfg" / "task" / "grading.example.json"
TASK_FIELDS = ("title", "summary")
LEVELS = ("L1", "L2", "L3", "L4")


def _grading() -> GradingConfig:
    return GradingConfig.from_path(GRADING)


def _corpus_mapping():
    return {
        c.category_id: c for c in load_corpus_categories(CORPUS)
    }


def _canonical_record(record_id: str, category_id: str, split: str | None, level: str | None) -> dict:
    leaf = category_id.split(":")[1]
    display = leaf.capitalize()
    record = {
        "id": record_id,
        "resolution_status": "resolved",
        "metadata": {"title": f"title {record_id}", "summary": f"summary {record_id}"},
        "classification": {"group": "G", "category": display},
        "data_level": level or "",
        "target": {
            "leaf_level": "category",
            "leaf_name": display,
            "category_id": category_id,
            "category_path": ["Synthetic", display],
        },
    }
    if split is not None:
        record["split"] = split
        record["split_exclusion_reason"] = None
    return record


def _seed(tmp_path: Path, records: list[dict]) -> Path:
    canonical = tmp_path / "canonical"
    canonical.mkdir(parents=True, exist_ok=True)
    (canonical / "all.json").write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    return canonical / "all.json"


def _balanced(level_for: dict[str, str] | None = None) -> list[dict]:
    categories = ["demo:alpha", "demo:bravo", "demo:charlie"]
    records = []
    for index, category in enumerate(categories):
        level = (level_for or {}).get(category, LEVELS[index % len(LEVELS)])
        records.append(_canonical_record(f"tr-{index}", category, "train", level))
        records.append(_canonical_record(f"va-{index}", category, "val", level))
        records.append(_canonical_record(f"te-{index}", category, "test", level))
    return records


def _export(tmp_path: Path, records: list[dict], name: str = "sft", **kwargs):
    canonical = _seed(tmp_path / name, records)
    return export_sft_dataset(
        canonical,
        None,
        tmp_path / name / "sft",
        LeafRegistry.from_path(REGISTRY),
        TaskConfig.from_mapping({"metadata_fields": TASK_FIELDS}),
        corpus=_corpus_mapping(),
        grading=_grading(),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# prompt / parser / answer shapes
# ---------------------------------------------------------------------------


def test_stage2_joint_prompt_contains_rubric_and_dual_key_contract() -> None:
    assets_registry = LeafRegistry.from_path(REGISTRY)
    config = TaskConfig.from_mapping({"metadata_fields": TASK_FIELDS})
    corpus = _corpus_mapping()
    cands = ["demo:charlie", "demo:alpha", "demo:bravo", "demo:delta", "demo:echo"]
    plain = build_stage2_prompt(
        {"title": "t"}, cands, assets_registry, config, corpus=corpus
    )
    joint = build_stage2_prompt(
        {"title": "t"}, cands, assets_registry, config, corpus=corpus,
        grading=_grading(),
    )
    assert '"answer"' in plain.system and "level" not in plain.system.split('"answer"')[1][:5]
    assert 'keys "answer" and "level"' in joint.system
    assert "Sensitivity levels:" in joint.user
    assert '[["L1","' in joint.user
    # rubric sits between bundle and metadata
    assert joint.user.index("Candidate bundle:") < joint.user.index("Sensitivity levels:") < joint.user.index("Field metadata:")
    # strict single-key contract unchanged when grading off
    assert parse_stage2_output('{"answer":"1"}').level is None
    with pytest.raises(Exception, match="must contain only answer"):
        parse_stage2_output('{"answer":"1","level":"L3"}')
    assert parse_stage2_output('{"answer":"1","level":"L3"}', allow_level=True).level == "L3"
    # assistant answer carries both keys only in joint mode
    assert json.loads(stage2_answer("demo:charlie", cands)) == {"answer": "1"}
    assert json.loads(stage2_answer("demo:charlie", cands, level="L3")) == {
        "answer": "1",
        "level": "L3",
    }


# ---------------------------------------------------------------------------
# exporter / validator / gate
# ---------------------------------------------------------------------------


def test_joint_export_rows_carry_level_and_validate(tmp_path) -> None:
    report = _export(tmp_path, _balanced())
    assert report["grading"] == {"enabled": True, "levels": list(LEVELS), "gt_field": "data_level"}
    assert report["label_gap_gate"]["status"] == "passed"

    import pyarrow.parquet as pq

    rows = pq.read_table(tmp_path / "sft" / "sft" / "train.parquet").to_pylist()
    stage2 = [r for r in rows if r["stage"] == "stage2"]
    assert stage2 and all(r["ground_truth_level"] for r in stage2)
    answers = [json.loads(r["messages"][-1]["content"]) for r in stage2]
    assert all(set(a) == {"answer", "level"} for a in answers)

    validation = validate_sft_dataset(
        tmp_path / "sft" / "sft",
        LeafRegistry.from_path(REGISTRY),
        TaskConfig.from_mapping({"metadata_fields": TASK_FIELDS}),
        _corpus_mapping(),
        grading=_grading(),
    )
    assert validation["valid"], validation["splits"]


def test_records_without_level_are_excluded_and_counted(tmp_path) -> None:
    records = _balanced()
    records.append(_canonical_record("tr-nolevel", "demo:delta", "train", None))
    report = _export(tmp_path, records, name="sft-nolevel")
    details = report["splits"]["train"]
    # the no-level record is excluded upstream with its own explicit counter
    assert details["split_records"] == 4
    assert details["exported_records"] == 6  # 3 records x 2 stages
    assert details["skipped_no_grading_label"] == 1
    assert details["skipped_not_resolved"] == 0
    assert report["trainable_resolved"] == 9  # only the balanced trio per split

    import pyarrow.parquet as pq

    rows = pq.read_table(
        tmp_path / "sft-nolevel" / "sft" / "train.parquet"
    ).to_pylist()
    assert all(r["source_id"] != "tr-nolevel" for r in rows)


def test_level_gap_blocks_export(tmp_path) -> None:
    records = _balanced()
    # L4 appears ONLY in test -> gate must fail
    records.append(_canonical_record("te-l4", "demo:delta", "test", "L4"))
    with pytest.raises(ValueError, match="label-gap gate failed"):
        _export(tmp_path, records)


def test_evaluation_reward_joint_semantics() -> None:
    registry = LeafRegistry.from_path(REGISTRY)
    cands = ["demo:charlie", "demo:alpha", "demo:bravo", "demo:delta", "demo:echo"]
    good = '{"answer":"1","level":"L3"}'
    bad_level = '{"answer":"1","level":"L9"}'
    missing_level = '{"answer":"1"}'

    ok = evaluate_stage2_choices(
        good, ground_truth="demo:charlie", candidates=cands, registry=registry,
        grading=_grading(), expected_level="L3",
    )
    assert ok.correct and ok.level_correct and ok.predicted_level == "L3"

    wrong_level = evaluate_stage2_choices(
        good.replace("L3", "L2"), ground_truth="demo:charlie", candidates=cands,
        registry=registry, grading=_grading(), expected_level="L3",
    )
    assert wrong_level.contract_valid and not wrong_level.correct
    assert not wrong_level.level_correct

    invalid = evaluate_stage2_choices(
        bad_level, ground_truth="demo:charlie", candidates=cands, registry=registry,
        grading=_grading(), expected_level="L3",
    )
    assert not invalid.contract_valid and any("L9" in e for e in invalid.errors)

    legacy_eval = evaluate_stage2_choices(
        missing_level, ground_truth="demo:charlie", candidates=cands, registry=registry,
    )
    assert legacy_eval.correct  # single-key path untouched

    reg_ids = tuple(cands)
    full = reward_stage2_choices(
        good, ground_truth="demo:charlie", candidates=reg_ids, registry=registry,
        grading=_grading(), expected_level="L3",
    )
    partial = reward_stage2_choices(
        good.replace("L3", "L2"), ground_truth="demo:charlie", candidates=reg_ids,
        registry=registry, grading=_grading(), expected_level="L3",
    )
    assert full.reward == 1.0 and partial.reward < 1.0 and "level" in partial.reason.lower()
