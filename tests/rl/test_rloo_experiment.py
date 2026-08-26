import hashlib
import json
from pathlib import Path

import pytest

from agent.task import GradingConfig, LeafRegistry
from script.verl.rl.rloo_experiment import (
    RlooExperimentConfig,
    build_validation_command,
    build_verl_command,
    compute_cascade_score,
    compute_score,
    main,
    write_run_manifest,
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
        "grading_manifest_path": (
            ROOT / "tests" / "sft" / "fixtures" / "grading_manifest.json"
        ),
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
    assert overrides["actor_rollout_ref.actor.use_kl_loss"] == "True"
    assert overrides["actor_rollout_ref.actor.kl_loss_coef"] == "0.001"
    assert overrides["algorithm.use_kl_in_reward"] == "False"
    assert overrides["actor_rollout_ref.rollout.mode"] == "async"
    assert overrides["actor_rollout_ref.rollout.multi_turn.enable"] == "True"
    assert (
        overrides["actor_rollout_ref.rollout.agent.default_agent_loop"]
        == "dataclassify_cascade"
    )
    assert Path(
        overrides["actor_rollout_ref.rollout.agent.agent_loop_config_path"]
    ).name == "cascade_agent_loop.yaml"
    assert overrides["reward.custom_reward_function.name"] == "compute_score"
    # pkg:// module form (server patch 2026-08-26): verl's load_module does not
    # register sys.modules for plain file paths, which crashes @dataclass
    # annotation resolution in the reward worker. The pkg:// branch requires
    # no .py suffix (ModuleNotFoundError otherwise).
    assert (
        overrides["reward.custom_reward_function.path"]
        == "pkg://script/verl/rl/rloo_experiment"
    )
    assert not any(argument.startswith("critic.") for argument in command)
    assert "grpo" not in " ".join(command).lower()


def test_formal_defaults_cover_two_turn_prompt_and_bridge(tmp_path: Path) -> None:
    config = _config(tmp_path)
    assert config.max_prompt_length >= 4096
    assert config.max_response_length >= 1024
    assert config.max_model_len >= 8192


def test_rloo_requires_multiple_sibling_rollouts(tmp_path: Path) -> None:
    config = _config(tmp_path, rollout_n=1)
    with pytest.raises(ValueError, match="rollout_n must be at least 2"):
        build_verl_command(config)


def test_formal_release_rejects_duplicate_dataset_names(tmp_path: Path) -> None:
    config = _config(tmp_path, dataset="finance+finance+shougang")
    with pytest.raises(ValueError, match="one finance\\+shougang"):
        config.validate_release_policy()


def test_validation_reuses_existing_rl_contract_cli(tmp_path: Path) -> None:
    config = _config(tmp_path)
    command = build_validation_command(config)

    assert command[:3] == ["python", "-m", "script.verl.rl.validate_cascade"]
    assert "--dataset-dir" in command
    assert "--registry" in command
    assert "--corpus" in command
    assert "--task-config" in command
    assert "--grading-manifest" in command
    assert "--report" in command
    assert Path(command[command.index("--report") + 1]) == config.validation_report_path
    assert "--metadata-fields" not in command


def test_run_manifest_binds_successful_validation_report_path_and_hash(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    report_path = config.validation_report_path
    report_path.parent.mkdir(parents=True)
    report_path.write_text(
        json.dumps({"format": "dataclassify-cascade-release-validation-v1", "valid": True}),
        encoding="utf-8",
    )
    manifest = write_run_manifest(
        config,
        {"passed": True, "gpu_validation_performed": False},
        build_validation_command(config),
        build_verl_command(config),
        validation_report_path=report_path,
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["validation_report"] == {
        "path": str(report_path.resolve()),
        "sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
    }


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


def test_static_reward_fallback_scores_direct_multiturn_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATACLASSIFY_RLOO_DATASET", "finance+shougang")
    monkeypatch.setenv("DATACLASSIFY_RLOO_REGISTRY", str(REGISTRY))
    monkeypatch.setenv(
        "DATACLASSIFY_RLOO_GRADING_MANIFEST",
        str(ROOT / "tests" / "sft" / "fixtures" / "grading_manifest.json"),
    )
    solution = (
        '{"candidates":["1","2","3","4","5"]}\n'
        "Stage 2 instructions: {\"field_name\":\"f\"}\n"
        '{"answer":"1","level":"L2"}'
    )
    assert compute_score(
        "finance/stage1",
        solution,
        "demo:alpha",
        {
            "dataset": "finance",
            "stage": "stage1",
            "ground_truth_level": "L2",
        },
    ) == pytest.approx(1.0)


def test_static_formal_stage2_reward_selects_dataset_grading_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATACLASSIFY_RLOO_DATASET", "finance+shougang")
    monkeypatch.setenv("DATACLASSIFY_RLOO_REGISTRY", str(REGISTRY))
    monkeypatch.setenv(
        "DATACLASSIFY_RLOO_GRADING_MANIFEST",
        str(ROOT / "tests" / "sft" / "fixtures" / "grading_manifest.json"),
    )

    candidates = [
        "demo:charlie",
        "demo:alpha",
        "demo:bravo",
        "demo:delta",
        "demo:echo",
    ]
    info = {
        "dataset": "finance",
        "stage": "stage2",
        "ground_truth_level": "L2",
        "candidates": candidates,
    }
    assert compute_score(
        "finance/stage2",
        '{"answer":"2","level":"L2"}',
        "demo:alpha",
        info,
    ) == 1.0
    assert compute_score(
        "finance/stage2",
        '{"answer":"2","level":"L1"}',
        "demo:alpha",
        info,
    ) == 0.5


def test_static_formal_stage2_reward_fails_closed_without_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATACLASSIFY_RLOO_DATASET", "finance+shougang")
    monkeypatch.setenv("DATACLASSIFY_RLOO_REGISTRY", str(REGISTRY))
    monkeypatch.delenv("DATACLASSIFY_RLOO_GRADING_MANIFEST", raising=False)
    with pytest.raises(RuntimeError, match="GRADING_MANIFEST"):
        compute_score(
            "shougang/stage2",
            '{"answer":"1","level":"L1"}',
            "demo:alpha",
            {
                "dataset": "shougang",
                "stage": "stage2",
                "ground_truth_level": "L1",
                "candidates": [
                    "demo:alpha",
                    "demo:bravo",
                    "demo:charlie",
                    "demo:delta",
                    "demo:echo",
                ],
            },
        )


def test_compute_cascade_score_requires_grading_and_fails_closed() -> None:
    registry = LeafRegistry.from_path(REGISTRY)
    assert compute_cascade_score(
        '{"candidates":["1","2","3","4","5"]}',
        '{"answer":"1","level":"L1"}',
        ground_truth="demo:alpha",
        ground_truth_level="L1",
        registry=registry,
    ) == 0.0
    assert compute_cascade_score(
        '{"candidates":["1","2","3","4","5"]}',
        '{"answer":"1","level":"L1"}',
        ground_truth="demo:alpha",
        ground_truth_level="L1",
        registry=registry,
        grading=GradingConfig(("L1", "L2")),
    ) == pytest.approx(1.0)


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
            "--grading-manifest",
            str(ROOT / "tests" / "sft" / "fixtures" / "grading_manifest.json"),
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
