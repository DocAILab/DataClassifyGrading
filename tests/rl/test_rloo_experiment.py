import json
from pathlib import Path

import pytest

from script.verl.rl.rloo_experiment import (
    RlooExperimentConfig,
    build_validation_command,
    build_verl_command,
    compute_score,
    main,
)


ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "cfg" / "task" / "leaf_registry.example.json"


def _config(tmp_path: Path, **overrides) -> RlooExperimentConfig:
    values = {
        "dataset": "demo",
        "model_path": tmp_path / "model",
        "data_dir": tmp_path / "rl-data",
        "registry_path": REGISTRY,
        "corpus_path": ROOT / "cfg" / "task" / "corpus.example.json",
        "task_config_path": ROOT / "cfg" / "task" / "task.example.json",
        "output_dir": tmp_path / "output",
        "python_bin": "python",
    }
    values.update(overrides)
    return RlooExperimentConfig(**values)


def _overrides(command: list[str]) -> dict[str, str]:
    return dict(argument.split("=", 1) for argument in command[3:])


def test_command_selects_rloo_without_changing_task_framework(tmp_path: Path) -> None:
    config = _config(tmp_path, rollout_n=3)
    command = build_verl_command(config)
    overrides = _overrides(command)

    assert command[:3] == ["python", "-m", "verl.trainer.main_ppo"]
    assert overrides["algorithm.adv_estimator"] == "rloo"
    assert overrides["actor_rollout_ref.rollout.n"] == "3"
    assert overrides["actor_rollout_ref.actor.use_kl_loss"] == "False"
    assert overrides["algorithm.use_kl_in_reward"] == "True"
    assert overrides["reward.custom_reward_function.name"] == "compute_score"
    assert Path(overrides["reward.custom_reward_function.path"]).name == "rloo_experiment.py"
    assert not any(argument.startswith("critic.") for argument in command)
    assert "grpo" not in " ".join(command).lower()


def test_rloo_requires_multiple_sibling_rollouts(tmp_path: Path) -> None:
    config = _config(tmp_path, rollout_n=1)
    with pytest.raises(ValueError, match="rollout_n must be at least 2"):
        build_verl_command(config)


def test_validation_reuses_existing_rl_contract_cli(tmp_path: Path) -> None:
    config = _config(tmp_path)
    command = build_validation_command(config)

    assert command[:3] == ["python", "-m", "script.verl.rl.validate"]
    assert "--dataset-dir" in command
    assert "--registry" in command
    assert "--corpus" in command
    assert "--task-config" in command
    fields = command[command.index("--metadata-fields") + 1 :]
    assert fields == ["title", "summary"]


def test_reward_entrypoint_routes_choice_outputs_through_shared_reward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATACLASSIFY_RLOO_DATASET", "demo")
    monkeypatch.setenv("DATACLASSIFY_RLOO_REGISTRY", str(REGISTRY))

    stage1 = compute_score(
        "demo/stage1",
        json.dumps({"candidates": ["3", "1", "2", "4", "5"]}),
        "demo:charlie",
        {"dataset": "demo", "stage": "stage1"},
    )
    stage2 = compute_score(
        "demo/stage2",
        json.dumps({"answer": "1"}),
        "demo:charlie",
        {
            "dataset": "demo",
            "stage": "stage2",
            "candidates": [
                "demo:charlie",
                "demo:alpha",
                "demo:bravo",
                "demo:delta",
                "demo:echo",
            ],
        },
    )

    assert stage1 == 1.0
    assert stage2 == 1.0


def test_reward_entrypoint_rejects_dataset_routing_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATACLASSIFY_RLOO_DATASET", "demo")
    monkeypatch.setenv("DATACLASSIFY_RLOO_REGISTRY", str(REGISTRY))

    with pytest.raises(ValueError, match="does not match configured dataset"):
        compute_score(
            "other/stage1",
            "not-json",
            "demo:charlie",
            {"dataset": "other", "stage": "stage1"},
        )


def test_dry_run_prints_commands_without_importing_or_running_verl(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    result = main(
        [
            "--dataset",
            "demo",
            "--model",
            str(tmp_path / "missing-model"),
            "--data-dir",
            str(tmp_path / "missing-data"),
            "--registry",
            str(REGISTRY),
            "--corpus",
            str(ROOT / "cfg" / "task" / "corpus.example.json"),
            "--task-config",
            str(ROOT / "cfg" / "task" / "task.example.json"),
            "--output-dir",
            str(tmp_path / "output"),
            "--dry-run",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert result == 0
    assert payload["algorithm"] == "rloo"
    assert payload["server_validation_performed"] is False
    assert "algorithm.adv_estimator=rloo" in payload["training_command"]
