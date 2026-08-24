"""Train a fresh DPO LoRA from an immutable merged SFT policy/reference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Sequence

from method.dpo.script.mine_preferences import adapter_identity
from method.dpo.training import (
    DPO_DEFAULTS,
    assert_separate_output,
    create_dpo_trainer,
    load_merged_sft_model,
    model_parameter_fingerprint,
    parameter_update_stats,
    snapshot_trainable_parameters,
    validate_runtime_versions,
)


def build_run_config(
    *,
    model: str,
    sft_adapter: str,
    preferences: str,
    output_dir: str | Path,
    max_steps: int,
    max_length: int,
    max_prompt_length: int,
) -> dict:
    return {
        "algorithm": "DPO",
        "stage": "stage2",
        "base_model": str(Path(model).resolve()),
        "sft_adapter": str(Path(sft_adapter).resolve()),
        "preferences": str(Path(preferences).resolve()),
        "output_dir": str(Path(output_dir).resolve()),
        "defaults": dict(DPO_DEFAULTS),
        "max_steps": max_steps,
        "max_length": max_length,
        "max_prompt_length": max_prompt_length,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--sft-adapter", required=True)
    parser.add_argument("--preferences", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-prompt-length", type=int, default=384)
    parser.add_argument("--resume-from-checkpoint")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    versions = validate_runtime_versions()
    assert_separate_output(args.sft_adapter, args.output_dir)
    preferences = Path(args.preferences)
    if not preferences.is_file():
        raise FileNotFoundError(f"preference parquet not found: {preferences}")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = build_run_config(
        model=args.model,
        sft_adapter=args.sft_adapter,
        preferences=args.preferences,
        output_dir=output,
        max_steps=args.max_steps,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
    )
    config["runtime_versions"] = versions
    config["sft_model_identity"] = adapter_identity(args.model, args.sft_adapter)
    (output / "run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = Dataset.from_parquet(str(preferences))
    if not dataset:
        raise ValueError("preference dataset must not be empty")
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    policy = load_merged_sft_model(args.model, args.sft_adapter)
    reference = load_merged_sft_model(args.model, args.sft_adapter)
    reference_before = model_parameter_fingerprint(reference)
    trainer = create_dpo_trainer(
        policy,
        reference,
        tokenizer,
        dataset,
        output,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        max_steps=args.max_steps,
    )
    before = snapshot_trainable_parameters(trainer.model)
    result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint or None)
    after = snapshot_trainable_parameters(trainer.model)
    policy_update = parameter_update_stats(before, after)
    reference_after = model_parameter_fingerprint(reference)
    final_adapter = output / "final_adapter"
    trainer.save_model(str(final_adapter))
    tokenizer.save_pretrained(final_adapter)
    metrics_path = output / "trainer_metrics.jsonl"
    with metrics_path.open("w", encoding="utf-8") as handle:
        for row in trainer.state.log_history:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    adapter_weights = final_adapter / "adapter_model.safetensors"
    report = {
        "algorithm": "DPO",
        "completed": True,
        "optimizer_steps": int(trainer.state.global_step),
        "train_metrics": result.metrics,
        "policy_update": policy_update,
        "reference_before": reference_before,
        "reference_after": reference_after,
        "reference_unchanged": reference_before == reference_after,
        "elapsed_seconds": time.monotonic() - started,
        "gpu_peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "final_adapter": str(final_adapter.resolve()),
        "final_adapter_sha256": _sha256(adapter_weights),
        "sft_model_identity": config["sft_model_identity"],
    }
    (output / "training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "COMPLETE").write_text("complete\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
