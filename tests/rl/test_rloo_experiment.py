import hashlib
import json
from pathlib import Path

import pytest

import script.verl.rl.rloo_experiment as rloo_module
from agent.task import GradingConfig, LeafRegistry
from script.verl.rl.rloo_experiment import (
    RlooExperimentConfig,
    build_validation_command,
    build_verl_command,
    compute_cascade_score,
    compute_score,
    main,
    validate_preflight,
    write_run_manifest,
)


ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "cfg" / "task" / "leaf_registry.example.json"
GRADING_MANIFEST = ROOT / "tests" / "rl" / "fixtures" / "grading_manifest.json"


def _config(tmp_path: Path, **overrides) -> RlooExperimentConfig:
    values = {
        "dataset": "shougang",
        "model_path": tmp_path / "model",
        "data_dir": tmp_path / "rl-data",
        "registry_path": REGISTRY,
        "corpus_path": ROOT / "cfg" / "task" / "corpus.example.json",
        "task_config_path": ROOT / "cfg" / "task" / "task.example.json",
        "output_dir": tmp_path / "output",
        "grading_manifest_path": GRADING_MANIFEST,
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
    # Live verl 0.9.0 Hydra rejects a no-+ override of this nested key with
    # ``Key 'save_lora_only' is not in struct (object_type=dict)``: the
    # actor.checkpoint node is declared as a dict in the structured config,
    # and struct mode requires "+" for new keys. The trainer CheckpointConfig
    # dataclass carries the field, and the checkpoint manager consumes the
    # "+"-injected value via DictConfig.get("save_lora_only", False); the
    # effective path is the nested one (engine_workers passes
    # actor_config.checkpoint).
    assert (
        overrides["+actor_rollout_ref.actor.checkpoint.save_lora_only"] == "True"
    )
    assert [
        argument
        for argument in command
        if argument.startswith("+") and "checkpoint" in argument
    ] == ["+actor_rollout_ref.actor.checkpoint.save_lora_only=True"]
    assert not any(argument.startswith("+checkpoint.") for argument in command)
    assert overrides["algorithm.use_kl_in_reward"] == "False"
    assert overrides["actor_rollout_ref.rollout.mode"] == "async"
    assert overrides["actor_rollout_ref.rollout.multi_turn.enable"] == "True"
    assert overrides["actor_rollout_ref.rollout.multi_turn.format"] == "qwen3_coder"
    assert Path(
        overrides["actor_rollout_ref.rollout.multi_turn.function_tool_path"]
    ).name == "native_tools.py"
    assert overrides["actor_rollout_ref.rollout.multi_turn.max_assistant_turns"] == "4"
    assert overrides["actor_rollout_ref.rollout.multi_turn.max_user_turns"] == "3"
    assert overrides["actor_rollout_ref.rollout.multi_turn.max_parallel_calls"] == "1"
    # Coordinator decision (2026-08-27): thinking is enabled by VeRL default;
    # the launcher must NOT emit an enable_thinking=False override. Think
    # stripping happens in the render layer; parser/reward are unchanged.
    assert "enable_thinking" not in " ".join(command)
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


def _cli_args(tmp_path: Path, *extra: str) -> list[str]:
    return [
        "--dataset",
        "shougang",
        "--model",
        str(tmp_path / "model"),
        "--data-dir",
        str(tmp_path / "rl-data"),
        "--registry",
        str(REGISTRY),
        "--corpus",
        str(ROOT / "cfg" / "task" / "corpus.example.json"),
        "--task-config",
        str(ROOT / "cfg" / "task" / "task.example.json"),
        "--grading-manifest",
        str(GRADING_MANIFEST),
        "--output-dir",
        str(tmp_path / "output"),
        *extra,
    ]


def test_direct_mode_supports_explicit_disabled_checkpoint(tmp_path: Path) -> None:
    config = _config(
        tmp_path, rollout_mode="direct", diagnostic_smoke=True, save_freq=-1
    )
    overrides = _overrides(build_verl_command(config))
    assert overrides["trainer.save_freq"] == "-1"
    # Nested checkpoint key must keep Hydra's "+" struct-add prefix: the
    # actor.checkpoint node is a dict in the structured config, and live
    # verl 0.9.0 rejects a no-+ override (object_type=dict).
    assert overrides["+actor_rollout_ref.actor.checkpoint.save_lora_only"] == "True"


def test_direct_static_view_supports_explicit_positive_checkpoint_frequency(
    tmp_path: Path,
) -> None:
    # Positive save_freq in direct mode is legal only for the static dry-run
    # view; a real direct launch is a diagnostic smoke and must disable
    # checkpoints (see test_diagnostic_smoke_requires_disabled_checkpoint).
    config = _config(tmp_path, rollout_mode="direct", dry_run=True, save_freq=2)
    overrides = _overrides(build_verl_command(config))
    assert overrides["trainer.save_freq"] == "2"
    assert overrides["+actor_rollout_ref.actor.checkpoint.save_lora_only"] == "True"


def test_tool_loop_rejects_nonpositive_save_frequency(tmp_path: Path) -> None:
    config = _config(tmp_path, rollout_mode="tool_loop", save_freq=-1)
    with pytest.raises(ValueError, match="save_freq"):
        build_verl_command(config)


def test_skip_reference_provenance_is_dry_run_only(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rejected = main(_cli_args(tmp_path, "--skip-reference-provenance"))
    assert rejected == 2
    assert "only allowed with --dry-run" in capsys.readouterr().err

    accepted = main(
        _cli_args(tmp_path, "--skip-reference-provenance", "--dry-run")
    )
    assert accepted == 0


def test_direct_preflight_summary_is_mode_aware(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference = tmp_path / "reference.provenance.json"
    reference.write_text("{}", encoding="utf-8")
    data_dir = tmp_path / "rl-data"
    for split in ("train", "val", "test"):
        (data_dir / f"{split}.parquet").parent.mkdir(parents=True, exist_ok=True)
        (data_dir / f"{split}.parquet").write_bytes(b"fixture")
    model_config = tmp_path / "model" / "config.json"
    model_config.parent.mkdir(parents=True)
    model_config.write_text("{}", encoding="utf-8")
    config = _config(
        tmp_path,
        rollout_mode="direct",
        diagnostic_smoke=True,
        save_freq=0,
        reference_provenance_path=reference,
        vllm_version="0.27.1",
    )
    monkeypatch.setattr(rloo_module, "_check_formal_assets", lambda _: {"ok": True})
    monkeypatch.setattr(
        rloo_module, "_check_reference_provenance", lambda _: {"verified": True}
    )
    monkeypatch.setattr(
        rloo_module, "_check_prompt_budget", lambda _: {"files": {}}
    )

    preflight = validate_preflight(config, check_runtime=False)
    trajectory = preflight["trajectory"]
    assert trajectory["rollout_mode"] == "direct"
    assert trajectory["max_tool_calls"] == 0
    assert trajectory["tools"] == []
    assert "search_top_k" not in trajectory


def test_direct_mode_builds_standard_single_turn_command_without_runtime_tool_config(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, rollout_mode="direct", dry_run=True)
    command = build_verl_command(config)
    overrides = _overrides(command)

    assert not any(
        argument.startswith("actor_rollout_ref.rollout.multi_turn.")
        or argument.startswith("actor_rollout_ref.rollout.agent.")
        or "function_tool_path" in argument
        or "agent_loop_config_path" in argument
        for argument in command
    )
    assert overrides["algorithm.adv_estimator"] == "rloo"
    assert overrides["actor_rollout_ref.rollout.n"] == str(config.rollout_n)
    assert overrides["actor_rollout_ref.rollout.mode"] == "async"
    assert not any(argument.startswith("rollout_mode=") for argument in command)
    assert overrides["reward.custom_reward_function.name"] == "compute_score"
    assert (
        overrides["reward.custom_reward_function.path"]
        == "pkg://script/verl/rl/rloo_experiment"
    )
    assert overrides["+actor_rollout_ref.actor.checkpoint.save_lora_only"] == "True"
    forbidden_fragments = (
        "function_tool_path",
        "tool_config_path",
        "agent_loop_config_path",
        "multi_turn",
        "default_agent_loop",
        "tools=",
        "tool_schema",
    )
    assert not any(
        any(fragment in argument for fragment in forbidden_fragments)
        for argument in command
    )


def test_explicit_tool_loop_mode_retains_native_tool_configuration(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, rollout_mode="tool_loop")
    command = build_verl_command(config)
    overrides = _overrides(command)

    assert overrides["actor_rollout_ref.rollout.multi_turn.enable"] == "True"
    assert overrides["actor_rollout_ref.rollout.multi_turn.format"] == "qwen3_coder"
    assert Path(
        overrides["actor_rollout_ref.rollout.multi_turn.function_tool_path"]
    ).name == "native_tools.py"
    assert overrides["actor_rollout_ref.rollout.agent.default_agent_loop"] == (
        "dataclassify_cascade"
    )
    assert Path(
        overrides["actor_rollout_ref.rollout.agent.agent_loop_config_path"]
    ).name == "cascade_agent_loop.yaml"
    assert overrides["+actor_rollout_ref.actor.checkpoint.save_lora_only"] == "True"


def test_invalid_rollout_mode_fails_fast(tmp_path: Path) -> None:
    config = _config(tmp_path, rollout_mode="unsupported")
    with pytest.raises(ValueError, match="rollout_mode"):
        build_verl_command(config)


def test_direct_real_launch_requires_diagnostic_smoke_opt_in(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = _config(tmp_path, rollout_mode="direct")
    with pytest.raises(ValueError, match="--diagnostic-smoke"):
        build_verl_command(config)
    rejected = main(_cli_args(tmp_path, "--rollout-mode", "direct"))
    assert rejected == 2
    assert "--diagnostic-smoke" in capsys.readouterr().err


def test_diagnostic_smoke_requires_direct_rollout_mode(tmp_path: Path) -> None:
    config = _config(tmp_path, diagnostic_smoke=True)
    with pytest.raises(ValueError, match="rollout_mode=direct"):
        build_verl_command(config)


def test_diagnostic_smoke_requires_disabled_checkpoint(tmp_path: Path) -> None:
    config = _config(
        tmp_path, rollout_mode="direct", diagnostic_smoke=True, save_freq=1
    )
    with pytest.raises(ValueError, match="save_freq <= 0"):
        build_verl_command(config)


def test_diagnostic_smoke_enforces_short_step_guard(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        rollout_mode="direct",
        diagnostic_smoke=True,
        save_freq=0,
        total_training_steps=4,
    )
    with pytest.raises(ValueError, match="total_training_steps"):
        build_verl_command(config)


def test_diagnostic_smoke_accepts_three_steps(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        rollout_mode="direct",
        diagnostic_smoke=True,
        save_freq=0,
        total_training_steps=3,
    )
    overrides = _overrides(build_verl_command(config))
    assert overrides["trainer.save_freq"] == "0"
    assert overrides["trainer.total_training_steps"] == "3"


def test_skip_reference_provenance_rejected_for_real_launches(
    tmp_path: Path,
) -> None:
    for kwargs in (
        {"rollout_mode": "tool_loop"},
        {"rollout_mode": "direct", "diagnostic_smoke": True, "save_freq": 0},
    ):
        config = _config(tmp_path, skip_reference_provenance=True, **kwargs)
        with pytest.raises(ValueError, match="only allowed with --dry-run"):
            build_verl_command(config)


def test_diagnostic_smoke_flag_is_launcher_only(tmp_path: Path) -> None:
    config = _config(
        tmp_path, rollout_mode="direct", diagnostic_smoke=True, save_freq=0
    )
    command = build_verl_command(config)
    # Assert on the override-key form, not a bare substring: pytest tmp dirs
    # are named after the test and can contain the flag name in file paths.
    assert not any("diagnostic_smoke=" in argument for argument in command)
    assert not any(
        argument.startswith(
            ("dry_run=", "rollout_mode=", "skip_reference_provenance=")
        )
        for argument in command
    )


def test_direct_dry_run_default_emits_disabled_checkpoint(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    result = main(_cli_args(tmp_path, "--rollout-mode", "direct", "--dry-run"))
    assert result == 0
    direct_payload = json.loads(capsys.readouterr().out)
    assert "trainer.save_freq=0" in direct_payload["training_command"]
    assert not any(
        "diagnostic_smoke=" in argument
        for argument in direct_payload["training_command"]
    )

    result = main(_cli_args(tmp_path, "--dry-run"))
    assert result == 0
    tool_loop_payload = json.loads(capsys.readouterr().out)
    assert "trainer.save_freq=1" in tool_loop_payload["training_command"]


def _preflight_fixture(tmp_path: Path) -> RlooExperimentConfig:
    reference = tmp_path / "reference.provenance.json"
    reference.write_text("{}", encoding="utf-8")
    data_dir = tmp_path / "rl-data"
    for split in ("train", "val", "test"):
        (data_dir / f"{split}.parquet").parent.mkdir(parents=True, exist_ok=True)
        (data_dir / f"{split}.parquet").write_bytes(b"fixture")
    model_config = tmp_path / "model" / "config.json"
    model_config.parent.mkdir(parents=True)
    model_config.write_text("{}", encoding="utf-8")
    return _config(
        tmp_path,
        reference_provenance_path=reference,
        vllm_version="0.27.1",
    )


def test_tool_loop_preflight_summary_reports_real_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _preflight_fixture(tmp_path)
    monkeypatch.setattr(rloo_module, "_check_formal_assets", lambda _: {"ok": True})
    monkeypatch.setattr(
        rloo_module, "_check_reference_provenance", lambda _: {"verified": True}
    )
    monkeypatch.setattr(
        rloo_module, "_check_prompt_budget", lambda _: {"files": {}}
    )

    preflight = validate_preflight(config, check_runtime=False)
    trajectory = preflight["trajectory"]
    assert trajectory["rollout_mode"] == "tool_loop"
    assert trajectory["formal"] is True
    assert trajectory["max_tool_calls"] == 3
    assert trajectory["tools"] == [
        "search_categories",
        "get_category_details",
        "get_category_examples",
    ]
    assert "browse_categories" not in trajectory["tools"]
    assert trajectory["search_top_k"] == 5
    assert trajectory["rollout_n"] == config.rollout_n


def test_formal_defaults_cover_native_tool_trajectory(tmp_path: Path) -> None:
    config = _config(tmp_path)
    assert config.rollout_mode == "tool_loop"
    assert config.max_prompt_length >= 4096
    assert config.max_response_length >= 1024
    assert config.max_model_len >= 8192


def test_rloo_requires_multiple_sibling_rollouts(tmp_path: Path) -> None:
    config = _config(tmp_path, rollout_n=1)
    with pytest.raises(ValueError, match="rollout_n must be at least 2"):
        build_verl_command(config)


def test_formal_release_requires_exact_shougang_dataset(tmp_path: Path) -> None:
    for dataset in ("finance", "finance+shougang", "shougang+shougang", " Shougang"):
        config = _config(tmp_path, dataset=dataset)
        with pytest.raises(ValueError, match="exactly.*shougang"):
            config.validate_options()


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


def test_reward_entrypoint_is_strict_native_joint_exact_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATACLASSIFY_RLOO_DATASET", "shougang")
    monkeypatch.setenv("DATACLASSIFY_RLOO_REGISTRY", str(REGISTRY))
    monkeypatch.setenv("DATACLASSIFY_RLOO_GRADING_MANIFEST", str(GRADING_MANIFEST))
    info = {
        "dataset": "shougang",
        "stage": "stage1",
        "ground_truth_level": "L1",
        "trajectory_format": "qwen3.5-native-tools-v2",
    }

    assert compute_score(
        "shougang/stage1",
        '{"answer":"3","level":"L1"}',
        "demo:charlie",
        info,
    ) == 1.0
    assert compute_score(
        "shougang/stage1",
        '{"answer":"3","level":"L2"}',
        "demo:charlie",
        info,
    ) == 0.0
    assert compute_score(
        "shougang/stage1",
        'result: {"answer":"3","level":"L1"}',
        "demo:charlie",
        info,
    ) == 0.0
    assert compute_score(
        "shougang/stage1",
        '{"candidates":["1","2","3","4","5"]}',
        "demo:charlie",
        info,
    ) == 0.0


def test_native_reward_fails_closed_without_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATACLASSIFY_RLOO_DATASET", "shougang")
    monkeypatch.setenv("DATACLASSIFY_RLOO_REGISTRY", str(REGISTRY))
    monkeypatch.delenv("DATACLASSIFY_RLOO_GRADING_MANIFEST", raising=False)
    with pytest.raises(RuntimeError, match="GRADING_MANIFEST"):
        compute_score(
            "shougang/stage1",
            '{"answer":"1","level":"L1"}',
            "demo:alpha",
            {
                "dataset": "shougang",
                "stage": "stage1",
                "ground_truth_level": "L1",
                "trajectory_format": "qwen3.5-native-tools-v2",
            },
        )


def test_compute_cascade_score_rejects_retired_manual_protocol() -> None:
    registry = LeafRegistry.from_path(REGISTRY)
    grading = GradingConfig(("L1", "L2"))
    assert compute_cascade_score(
        '{"candidates":["1","2","3","4","5"]}',
        '{"answer":"1","level":"L1"}',
        ground_truth="demo:alpha",
        ground_truth_level="L1",
        registry=registry,
        grading=grading,
    ) == 0.0
    assert compute_cascade_score(
        "",
        '{"answer":"1","level":"L1"}',
        ground_truth="demo:alpha",
        ground_truth_level="L1",
        registry=registry,
        grading=grading,
    ) == 1.0


def test_reward_entrypoint_rejects_dataset_routing_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATACLASSIFY_RLOO_DATASET", "shougang")
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
            "shougang",
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
            str(GRADING_MANIFEST),
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
