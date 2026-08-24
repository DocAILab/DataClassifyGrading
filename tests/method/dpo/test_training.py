from pathlib import Path

import pytest
import torch

from method.dpo.training import (
    DPO_DEFAULTS,
    assert_separate_output,
    effective_save_steps,
    freeze_reference,
    normalize_trl_availability_flags,
    parameter_update_stats,
    validate_runtime_versions,
)
from method.dpo.script.train import build_run_config


def test_dpo_defaults_match_the_approved_experiment_contract():
    assert DPO_DEFAULTS == {
        "beta": 0.1,
        "learning_rate": 5e-7,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "num_train_epochs": 1.0,
        "bf16": True,
        "gradient_checkpointing": True,
        "save_steps": 100,
        "seed": 42,
    }


def test_smoke_saves_checkpoint_after_its_only_step():
    assert effective_save_steps(1) == 1
    assert effective_save_steps(-1) == 100
    assert effective_save_steps(500) == 100


def test_assert_separate_output_rejects_sft_directory_or_descendant(tmp_path):
    sft = tmp_path / "sft_adapter"
    sft.mkdir()
    assert_separate_output(sft, tmp_path / "dpo")
    with pytest.raises(ValueError, match="SFT"):
        assert_separate_output(sft, sft)
    with pytest.raises(ValueError, match="SFT"):
        assert_separate_output(sft, sft / "child")


def test_freeze_reference_disables_gradients_and_sets_eval():
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))
    model.train()
    freeze_reference(model)

    assert model.training is False
    assert all(parameter.requires_grad is False for parameter in model.parameters())


def test_parameter_update_stats_detects_policy_change_and_unchanged_reference():
    before = {"weight": torch.tensor([1.0, 2.0])}
    changed = {"weight": torch.tensor([1.0, 2.5])}
    unchanged = {"weight": torch.tensor([1.0, 2.0])}

    policy = parameter_update_stats(before, changed)
    reference = parameter_update_stats(before, unchanged)

    assert policy["absolute_update_norm"] == pytest.approx(0.5)
    assert policy["relative_update_norm"] > 0
    assert reference["absolute_update_norm"] == 0.0
    assert reference["relative_update_norm"] == 0.0


def test_parameter_update_stats_rejects_mismatched_parameter_names():
    with pytest.raises(ValueError, match="names"):
        parameter_update_stats({"a": torch.tensor([1.0])}, {"b": torch.tensor([1.0])})


def test_validate_runtime_versions_is_explicit_and_fail_closed():
    versions = {
        "trl": "0.24.0",
        "peft": "0.20.0",
        "transformers": "5.15.0",
        "datasets": "5.0.1",
    }
    assert validate_runtime_versions(versions.__getitem__) == versions

    versions["trl"] = "0.23.0"
    with pytest.raises(RuntimeError, match="trl"):
        validate_runtime_versions(versions.__getitem__)


def test_normalize_trl_availability_flags_converts_only_private_tuples():
    class Flags:
        _weave_available = (False, None)
        _peft_available = (True, "0.20.0")
        public_available = (False, None)
        _already_available = True

    changed = normalize_trl_availability_flags(Flags)

    assert changed == {"_peft_available": True, "_weave_available": False}
    assert Flags._weave_available is False
    assert Flags._peft_available is True
    assert Flags.public_available == (False, None)
    assert Flags._already_available is True


def test_build_run_config_records_smoke_and_full_overrides(tmp_path):
    config = build_run_config(
        model="/models/qwen",
        sft_adapter="/artifacts/sft/lora",
        preferences="/data/preferences.parquet",
        output_dir=tmp_path / "dpo",
        max_steps=1,
        max_length=640,
        max_prompt_length=512,
    )

    assert config["algorithm"] == "DPO"
    assert config["defaults"] == DPO_DEFAULTS
    assert config["max_steps"] == 1
    assert config["max_length"] == 640
    assert config["max_prompt_length"] == 512
    assert config["output_dir"] == str((tmp_path / "dpo").resolve())
