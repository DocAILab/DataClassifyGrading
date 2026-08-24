import json
from pathlib import Path

import pytest

from method.dpo.storage import assert_training_capacity, write_cleanup_report


def test_assert_training_capacity_requires_persistent_and_shm_headroom(monkeypatch):
    values = {Path("/data"): 21 * 2**30, Path("/dev/shm"): 36 * 2**30}
    monkeypatch.setattr("method.dpo.storage.free_bytes", lambda path: values[Path(path)])

    report = assert_training_capacity("/data", "/dev/shm")

    assert report["valid"] is True
    assert report["minimum_persistent_free_bytes"] == 20 * 2**30
    assert report["minimum_shm_free_bytes"] == 35 * 2**30

    values[Path("/data")] = 19 * 2**30
    with pytest.raises(RuntimeError, match="persistent"):
        assert_training_capacity("/data", "/dev/shm")


def test_write_cleanup_report_records_exact_files_and_reclaimed_bytes(tmp_path):
    output = tmp_path / "cleanup_report.json"
    report = write_cleanup_report(
        output,
        [
            {
                "path": "/artifacts/old/model.safetensors",
                "bytes": 15_231_272_152,
                "recoverable_from": "base model + permanent LoRA",
            }
        ],
    )

    assert report["reclaimed_bytes"] == 15_231_272_152
    assert json.loads(output.read_text(encoding="utf-8")) == report


def test_pipeline_is_detached_recoverable_and_never_mentions_test_json():
    root = Path(__file__).parents[3]
    runner = (root / "src/method/dpo/script/run_stage2_hard_dpo.sh").read_text(encoding="utf-8")
    starter = (root / "src/method/dpo/script/start_stage2_hard_dpo.sh").read_text(encoding="utf-8")

    assert "test.json" not in runner
    assert "SMOKE_VERIFIED" in runner
    assert "comparison_to_sft.json" in runner
    assert "checkpoint-" in runner
    assert "status.json" in runner
    assert "setsid" in starter
    assert "nohup" in starter
    assert "PPID" not in starter
