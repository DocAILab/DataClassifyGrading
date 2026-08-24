from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.task import GradingConfig, LeafRegistry
from agent.task.assets import load_corpus_categories
from agent.training.rl.cascade import (
    CascadeRunner,
    build_cascade_stage1_prompt,
    decode_stage2_output,
    leave_one_out_advantages,
    trajectory_to_policy_records,
)


ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "cfg" / "task" / "leaf_registry.example.json"
CORPUS = ROOT / "cfg" / "task" / "corpus.example.json"


@pytest.fixture
def assets():
    registry = LeafRegistry.from_path(REGISTRY)
    corpus = {entry.category_id: entry for entry in load_corpus_categories(CORPUS)}
    return registry, corpus, GradingConfig(("L1", "L2", "L3", "L4"))


class FakeRollout:
    def __init__(self, stage1: list[str], stage2: list[str]):
        self.stage1 = stage1
        self.stage2 = stage2
        self.stage2_prompts = []

    def rollout_stage1(self, prompt, *, n, source):
        assert n == 4
        return self.stage1

    def rollout_stage2(self, prompt, *, candidates, source):
        self.stage2_prompts.append((prompt, tuple(candidates)))
        return self.stage2[len(self.stage2_prompts) - 1]


def _record(category: str = "demo:alpha") -> dict:
    return {
        "id": "field-1",
        "source_group": "source-group-1",
        "field_name": "phone_number",
        "standards": "must not be rendered as free-form metadata",
        "target": {"category_id": category},
        "data_level": "L2",
    }


def test_formal_stage2_shape_is_strict_and_shared() -> None:
    candidates = ("demo:alpha", "demo:bravo", "demo:charlie", "demo:delta", "demo:echo")
    assert decode_stage2_output(
        '{"answer":"1","level":"L2"}', candidates=candidates,
        grading=GradingConfig(("L1", "L2")),
    )[:2] == ("demo:alpha", "L2")
    leaf_shape = decode_stage2_output(
        '{"leaf":"1","data_level":"L2"}', candidates=candidates,
        grading=GradingConfig(("L1", "L2")),
    )
    assert leaf_shape[2]


def test_direct_cascade_runs_stage2_on_valid_miss_and_uses_actual_candidates(assets) -> None:
    registry, corpus, grading = assets
    # Candidate choices 1..5 decode to alpha..echo; target foxtrot is a valid
    # miss and must not be injected into the Stage 2 prompt.
    fake = FakeRollout(
        ['{"candidates":["1","2","3","4","5"]}'] * 4,
        ['{"answer":"1","level":"L2"}'] * 4,
    )
    run = CascadeRunner(registry=registry, corpus=corpus, grading=grading, rollout=fake).run(
        [_record("demo:foxtrot")]
    )
    assert len(run.trajectories) == 4
    assert all(item.stage1.valid and not item.stage1.hit for item in run.trajectories)
    assert all(item.stage2 is not None for item in run.trajectories)
    assert all(item.stage2_candidates == tuple(corpus)[:5] for item in run.trajectories)
    assert all(item.reward == pytest.approx(0.3) for item in run.trajectories)
    assert all("standards" not in prompt.user for prompt, _ in fake.stage2_prompts)


def test_invalid_stage1_terminates_without_stage2_and_still_has_two_segments(assets) -> None:
    registry, corpus, grading = assets
    fake = FakeRollout(['not-json'] * 4, [])
    run = CascadeRunner(registry=registry, corpus=corpus, grading=grading, rollout=fake).run(
        [_record()]
    )
    assert all(item.terminated and item.reward == 0 for item in run.trajectories)
    assert not fake.stage2_prompts
    assert all(len(item.segments) == 2 for item in run.trajectories)
    assert all(item.segments[1].loss_mask == () for item in run.trajectories)


def test_formal_cascade_rejects_empty_field_name(assets) -> None:
    registry, corpus, grading = assets
    fake = FakeRollout(['{"candidates":["1","2","3","4","5"]}'] * 4, [])
    record = _record()
    record["field_name"] = "   "
    with pytest.raises(ValueError, match="field_name"):
        CascadeRunner(
            registry=registry, corpus=corpus, grading=grading, rollout=fake
        ).run([record])


def test_formal_cascade_requires_grading_and_level_labels(assets) -> None:
    registry, corpus, grading = assets
    fake = FakeRollout(['{"candidates":["1","2","3","4","5"]}'] * 4, [])
    with pytest.raises(ValueError, match="grading"):
        CascadeRunner(registry=registry, corpus=corpus, rollout=fake)
    record = _record()
    record.pop("data_level")
    with pytest.raises(ValueError, match="ground_truth_level"):
        CascadeRunner(
            registry=registry, corpus=corpus, grading=grading, rollout=fake
        ).run([record])


def test_rloo_groups_by_original_source_id_not_shared_metadata_group(assets) -> None:
    registry, corpus, grading = assets
    fake = FakeRollout(
        ['{"candidates":["1","2","3","4","5"]}'] * 4,
        ['{"answer":"1","level":"L2"}'] * 8,
    )
    first = _record("demo:alpha")
    first["id"] = "source-a"
    first["source_group"] = "same-table"
    second = _record("demo:foxtrot")
    second["id"] = "source-b"
    second["source_group"] = "same-table"
    run = CascadeRunner(
        registry=registry, corpus=corpus, grading=grading, rollout=fake
    ).run([first, second])
    assert set(run.by_group()) == {"source-a", "source-b"}
    assert all(len(group) == 4 for group in run.by_group().values())
    assert run.advantages == pytest.approx((0.0,) * 8)


def test_rloo_groups_and_shared_segment_advantage() -> None:
    values = leave_one_out_advantages((1.0, 0.0, 0.5, 0.5), ("a", "a", "b", "b"))
    assert values == pytest.approx((1.0, -1.0, 0.0, 0.0))
