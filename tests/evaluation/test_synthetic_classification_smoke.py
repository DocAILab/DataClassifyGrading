import json
from pathlib import Path

from agent.task import ClassificationAssets
from script.verl.sft.evaluate_true_e2e import aggregate_true_e2e, run_one


ROOT = Path(__file__).resolve().parents[2]


def test_two_stage_classification_runs_with_synthetic_local_assets() -> None:
    assets = ClassificationAssets.from_files(
        registry=ROOT / "cfg" / "task" / "leaf_registry.example.json",
        corpus=ROOT / "cfg" / "task" / "corpus.example.json",
        task=ROOT / "cfg" / "task" / "task.example.json",
    )
    completions = iter(
        [
            json.dumps({"candidates": ["3", "1", "2", "4", "5"]}),
            json.dumps({"answer": "1"}),
        ]
    )

    outcome = run_one(
        {
            "source_id": "synthetic-row-1",
            "metadata": {"title": "fabricated title", "summary": "fabricated summary"},
            "ground_truth": "demo:charlie",
        },
        registry=assets.registry,
        config=assets.task,
        corpus=assets.corpus,
        seed=1,
        generate=lambda _messages: next(completions),
    )

    assert outcome.predicted_top5 == (
        "demo:charlie",
        "demo:alpha",
        "demo:bravo",
        "demo:delta",
        "demo:echo",
    )
    assert outcome.final_decision == "demo:charlie"
    assert outcome.e2e_correct is True
    assert aggregate_true_e2e([outcome])["true_e2e_accuracy"] == 1.0
