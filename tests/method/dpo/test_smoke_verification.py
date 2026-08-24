import json

import pytest

from method.dpo.smoke_verification import verify_smoke


def _write_valid_smoke(root):
    training = root / "training"
    training.mkdir()
    (training / "training_report.json").write_text(
        json.dumps(
            {
                "completed": True,
                "optimizer_steps": 1,
                "train_metrics": {"train_loss": 0.8},
                "policy_update": {
                    "absolute_update_norm": 0.02,
                    "relative_update_norm": 0.001,
                },
                "reference_unchanged": True,
            }
        ),
        encoding="utf-8",
    )
    (training / "trainer_metrics.jsonl").write_text(
        json.dumps({"loss": 0.8, "grad_norm": 1.2, "learning_rate": 5e-7}) + "\n",
        encoding="utf-8",
    )
    adapter = training / "final_adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    checkpoint = training / "checkpoint-1"
    checkpoint.mkdir()
    (checkpoint / "trainer_state.json").write_text("{}", encoding="utf-8")
    return training


def test_verify_smoke_requires_real_update_metrics_adapter_and_checkpoint(tmp_path):
    training = _write_valid_smoke(tmp_path)
    report = verify_smoke(training, tmp_path / "verification")

    assert report["valid"] is True
    assert report["optimizer_steps"] == 1
    assert report["policy_absolute_update_norm"] == pytest.approx(0.02)
    assert (tmp_path / "verification" / "smoke_report.json").is_file()
    assert (tmp_path / "verification" / "SMOKE_VERIFIED").read_text() == "verified\n"


@pytest.mark.parametrize(
    "mutation,expected",
    [
        (lambda report: report.update({"reference_unchanged": False}), "reference"),
        (lambda report: report["policy_update"].update({"absolute_update_norm": 0.0}), "policy"),
        (lambda report: report["train_metrics"].update({"train_loss": float("nan")}), "loss"),
    ],
)
def test_verify_smoke_fails_closed_for_invalid_training_report(tmp_path, mutation, expected):
    training = _write_valid_smoke(tmp_path)
    path = training / "training_report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    mutation(report)
    path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match=expected):
        verify_smoke(training, tmp_path / "verification")
    assert not (tmp_path / "verification" / "SMOKE_VERIFIED").exists()


def test_verify_smoke_requires_finite_grad_norm_and_checkpoint(tmp_path):
    training = _write_valid_smoke(tmp_path)
    (training / "trainer_metrics.jsonl").write_text(
        json.dumps({"loss": 0.8, "grad_norm": None}) + "\n", encoding="utf-8"
    )
    for child in (training / "checkpoint-1").iterdir():
        child.unlink()
    (training / "checkpoint-1").rmdir()

    with pytest.raises(ValueError, match="grad_norm"):
        verify_smoke(training, tmp_path / "verification")
    assert not (tmp_path / "verification" / "SMOKE_VERIFIED").exists()
