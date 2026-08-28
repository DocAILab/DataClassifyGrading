from pathlib import Path

from method.retrieval.script.evaluate_stage1 import build_parser, model_identity


ROOT = Path(__file__).resolve().parents[3]
PIPELINE = ROOT / "src" / "method" / "retrieval" / "script" / "run_stage1_bge_m3.sh"
STARTER = ROOT / "src" / "method" / "retrieval" / "script" / "start_stage1_bge_m3.sh"


def test_cli_has_no_split_or_test_option():
    options = {action.dest for action in build_parser()._actions}
    assert "split" not in options
    assert "splits" not in options
    assert {"input_dir", "registry", "model", "output_dir", "batch_size", "limit"} <= options


def test_cli_exposes_hybrid_fusion_weights():
    options = {action.dest for action in build_parser()._actions}
    assert {"lexical_weight", "dense_weight", "index_weight"} <= options


def test_ablation_grid_covers_index_dominant_weight_variants_without_test_split():
    grid = ROOT / "src" / "method" / "retrieval" / "script" / "run_hybrid_ablation_grid.sh"
    source = grid.read_text(encoding="utf-8")
    assert "index_dominant" in source
    assert "index_bge" in source
    assert "index_char" in source
    assert "test.json" not in source


def test_pipeline_runs_real_smoke_before_full_and_never_names_test_split():
    launcher = PIPELINE.read_text(encoding="utf-8")
    assert "--limit 16" in launcher
    assert "stage1-bge-m3-v1" in launcher
    assert "test.json" not in launcher
    assert launcher.index("--limit 16") < launcher.index("FULL_COMPLETE")
    assert "pytest" in launcher
    starter = STARTER.read_text(encoding="utf-8")
    assert "setsid" in starter and "nohup" in starter


def test_model_identity_hashes_transformer_config_and_weights(tmp_path):
    (tmp_path / "config.json").write_text("config", encoding="utf-8")
    (tmp_path / "pytorch_model.bin").write_bytes(b"weights")
    first = model_identity(tmp_path)
    assert first.startswith("sha256:")
    (tmp_path / "pytorch_model.bin").write_bytes(b"changed")
    assert model_identity(tmp_path) != first
