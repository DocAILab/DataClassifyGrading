"""Synthetic end-to-end evaluator tests.

Drives ``evaluate_true_e2e.run_one`` / ``aggregate_true_e2e`` on CPU with an
injected generator. Covers:

- Stage1 miss  -> true E2E fails (GT is not among the PREDICTED top-5)
- Stage1 hit + Stage2 correct -> true E2E success
- Stage1 hit + Stage2 wrong   -> true E2E fail (conditional denominator counts)
- malformed Stage1 / Stage2   -> never raises, maps to zero
- Stage2 candidates must be EXACTLY the Stage1 predicted top-5

Identity, decoding, and prompt behavior use the shared task interfaces.
"""

from __future__ import annotations

import json

import pytest

from agent.task.contracts import CorpusCategory, LeafRegistry, TaskConfig
from script.verl.sft.evaluate_true_e2e import aggregate_true_e2e, run_one

# ---- synthetic registry/corpus: 8 categories -> choice ids "1".."8" ----
REG_IDS = [f"reg:{i}" for i in range(1, 9)]


def make_registry() -> LeafRegistry:
    return LeafRegistry.from_mapping(REG_IDS)


def make_corpus() -> dict[str, CorpusCategory]:
    return {
        category_id: CorpusCategory(
            category_id=category_id,
            name=f"name-{category_id}",
            description=f"description-{category_id}",
            descriptions=(f"extra-{category_id}",),
            examples=(f"example-{category_id}",),
        )
        for category_id in REG_IDS
    }


def make_config() -> TaskConfig:
    return TaskConfig(metadata_fields=("field_name", "field_description"))


def source_row(gt: str = "reg:3", source_id: str = "s-1") -> dict:
    return {
        "stage": "stage1",
        "source_id": source_id,
        "metadata": {"field_name": "fn", "field_description": "fd"},
        "ground_truth": gt,
    }


def make_generate(stage1_out: str, stage2_out: str):
    """Injected generate: dispatch on the prompt type (Stage 1 catalog vs bundle)."""

    def generate(messages):
        user_content = messages[1]["content"]
        if "Candidate bundle:" in user_content:
            return stage2_out
        return stage1_out

    return generate


@pytest.fixture
def ctx():
    return {
        "registry": make_registry(),
        "config": make_config(),
        "corpus": make_corpus(),
    }


def test_stage1_hit_and_stage2_correct_true_e2e(ctx):
    row = source_row(gt="reg:3")
    gen = make_generate(
        stage1_out=json.dumps({"candidates": ["3", "1", "2", "4", "5"]}),
        stage2_out=json.dumps({"answer": "1"}),  # predicted[0] == reg:3 == GT
    )
    out = run_one(row, seed=1, generate=gen, **ctx)
    assert out.stage1_format_valid and out.stage1_contract_valid
    assert out.recalled is True
    assert out.predicted_top5 == ("reg:3", "reg:1", "reg:2", "reg:4", "reg:5")
    assert out.stage2_attempted
    assert out.stage2_prompt_candidates == out.predicted_top5
    assert out.stage2_correct and out.final_decision == "reg:3"
    assert out.e2e_correct is True
    metrics = aggregate_true_e2e([out])
    assert metrics["stage1_recall_at_5"] == 1.0
    assert metrics["stage2_conditional_accuracy"] == 1.0
    assert metrics["true_e2e_accuracy"] == 1.0


def test_stage1_hit_and_stage2_wrong_true_e2e_fails(ctx):
    row = source_row(gt="reg:3")
    gen = make_generate(
        stage1_out=json.dumps({"candidates": ["3", "1", "2", "4", "5"]}),
        stage2_out=json.dumps({"answer": "2"}),  # predicted[1] == reg:1 != GT
    )
    out = run_one(row, seed=1, generate=gen, **ctx)
    assert out.recalled is True
    assert out.stage2_attempted and not out.stage2_correct
    assert out.e2e_correct is False
    metrics = aggregate_true_e2e([out])
    assert metrics["stage1_recall_at_5"] == 1.0
    # conditional accuracy: denominator = recalled (1), numerator = correct (0)
    assert metrics["stage2_conditional_accuracy"] == 0.0
    assert metrics["true_e2e_accuracy"] == 0.0


def test_stage1_miss_stage2_still_uses_predicted_top5_and_fails(ctx):
    row = source_row(gt="reg:3")
    # contract-valid top-5 that does NOT contain the GT (reg:3)
    gen = make_generate(
        stage1_out=json.dumps({"candidates": ["1", "2", "4", "5", "6"]}),
        stage2_out=json.dumps({"answer": "1"}),
    )
    out = run_one(row, seed=1, generate=gen, **ctx)
    assert out.stage1_contract_valid
    assert out.recalled is False  # GT not among predicted
    # Stage 2 is still built from the PREDICTED top-5 (never from the GT bundle)
    assert out.stage2_attempted
    assert out.stage2_prompt_candidates == ("reg:1", "reg:2", "reg:4", "reg:5", "reg:6")
    assert out.predicted_top5 == ("reg:1", "reg:2", "reg:4", "reg:5", "reg:6")
    assert out.e2e_correct is False
    metrics = aggregate_true_e2e([out])
    assert metrics["stage1_recall_at_5"] == 0.0
    # conditional accuracy excludes (denominator) the recalled==False source
    assert metrics["stage2_conditional_accuracy"] == 0.0
    assert metrics["true_e2e_accuracy"] == 0.0


def test_malformed_stage1_never_raises(ctx):
    row = source_row(gt="reg:3")
    gen = make_generate(stage1_out="not json at all", stage2_out="unused")
    out = run_one(row, seed=1, generate=gen, **ctx)
    assert not out.stage1_format_valid
    assert not out.stage1_contract_valid
    assert not out.recalled
    assert not out.stage2_attempted  # never prompts Stage 2 without a valid top-5
    assert out.e2e_correct is False
    assert out.failures  # structured error, not an exception


def test_malformed_stage2_never_raises(ctx):
    row = source_row(gt="reg:3")
    gen = make_generate(
        stage1_out=json.dumps({"candidates": ["3", "1", "2", "4", "5"]}),
        stage2_out="also not json",
    )
    out = run_one(row, seed=1, generate=gen, **ctx)
    assert out.recalled and out.stage2_attempted
    assert not out.stage2_format_valid and not out.stage2_contract_valid
    assert out.e2e_correct is False
    assert out.failures
    metrics = aggregate_true_e2e([out])
    assert metrics["stage2_format_failure_rate"] == 1.0
    assert metrics["stage2_contract_failure_rate"] == 1.0


def test_stage2_candidates_exactly_equal_predicted_top5(ctx):
    """The dynamic Stage-2 prompt must use the Stage-1 PREDICTED top-5, not the
    pre-constructed gold bundle (this is the definitional difference from the
    factorized/proxy evaluator)."""
    row = source_row(gt="reg:3")
    predicted = ["3", "1", "2", "4", "5"]
    captured = {}

    def generate(messages):
        user_content = messages[1]["content"]
        if "Candidate bundle:" in user_content:
            captured["stage2_user"] = user_content
            return json.dumps({"answer": "1"})
        return json.dumps({"candidates": predicted})

    out = run_one(row, seed=1, generate=generate, **ctx)
    assert out.stage2_prompt_candidates == out.predicted_top5 == (
        "reg:3", "reg:1", "reg:2", "reg:4", "reg:5",
    )
    bundle = json.loads(captured["stage2_user"].split("Candidate bundle:\n", 1)[1].split("\nField metadata:")[0])
    entry_ids = [entry["id"] for entry in bundle]
    # bundle ids are the LOCAL positions over the PREDICTED top-5
    assert entry_ids == ["1", "2", "3", "4", "5"]
    # names are the registry DISPLAY names (like the exported parquet),
    # descriptions/examples come from the canonical corpus by category_id
    names = [entry["name"] for entry in bundle]
    assert names == list(out.predicted_top5)
    assert bundle[0]["description"] == "description-reg:3"
    assert "extra-reg:3" in bundle[0]["descriptions"]
    assert "example-reg:3" in bundle[0]["examples"]


def test_aggregate_separates_conditional_and_true_e2e(ctx):
    gen_hit_correct = make_generate(
        stage1_out=json.dumps({"candidates": ["3", "1", "2", "4", "5"]}),
        stage2_out=json.dumps({"answer": "1"}),
    )
    gen_hit_wrong = make_generate(
        stage1_out=json.dumps({"candidates": ["3", "1", "2", "4", "5"]}),
        stage2_out=json.dumps({"answer": "2"}),
    )
    gen_miss = make_generate(
        stage1_out=json.dumps({"candidates": ["1", "2", "4", "5", "6"]}),
        stage2_out=json.dumps({"answer": "1"}),
    )
    outcomes = [
        run_one(source_row(gt="reg:3", source_id="a"), seed=1, generate=gen_hit_correct, **ctx),
        run_one(source_row(gt="reg:3", source_id="b"), seed=1, generate=gen_hit_wrong, **ctx),
        run_one(source_row(gt="reg:3", source_id="c"), seed=1, generate=gen_miss, **ctx),
    ]
    metrics = aggregate_true_e2e(outcomes)
    assert metrics["sources"] == 3
    assert metrics["stage1_recalled_count"] == 2
    assert metrics["stage2_conditional_accuracy"] == pytest.approx(1 / 2)  # 1 of 2 recalled correct
    assert metrics["true_e2e_accuracy"] == pytest.approx(1 / 3)  # 1 of 3 sources e2e correct
