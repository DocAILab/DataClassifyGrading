"""Mine SFT-scored semantic hard negatives from train.json only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

from agent.task import LeafRegistry, TaskConfig
from method.dpo.label_scoring import mine_score_rows, score_candidate_answers
from method.dpo.preference_data import load_train_records


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def adapter_identity(model_path: str | Path, adapter_path: str | Path) -> str:
    """Hash model identity plus immutable PEFT config and weights."""
    adapter = Path(adapter_path)
    files = [adapter / "adapter_config.json", adapter / "adapter_model.safetensors"]
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing SFT adapter files: {missing}")
    digest = hashlib.sha256(str(Path(model_path)).encode("utf-8"))
    for path in files:
        digest.update(path.name.encode("utf-8"))
        digest.update(_file_sha256(path).encode("ascii"))
    return f"sha256:{digest.hexdigest()}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--task-config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--sft-adapter", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    registry = LeafRegistry.from_path(args.registry)
    config = TaskConfig.from_path(args.task_config)
    if config.metadata_fields != ("field_name",):
        raise ValueError("mining requires metadata_fields=[field_name]")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model_identity = adapter_identity(args.model, args.sft_adapter)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        local_files_only=True,
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(base, args.sft_adapter, is_trainable=False)
    model.eval()

    def score_fn(prompt, answers):
        return score_candidate_answers(
            model,
            tokenizer,
            prompt,
            answers,
            batch_size=args.batch_size,
            device="cuda",
        )

    scores_path = output / "label_scores.jsonl"
    report = mine_score_rows(
        load_train_records(args.input_dir),
        registry,
        scores_path,
        score_fn=score_fn,
        model_identity=model_identity,
        seed=args.seed,
    )
    report.update(
        {
            "model_path": str(Path(args.model).resolve()),
            "sft_adapter_path": str(Path(args.sft_adapter).resolve()),
            "batch_size": args.batch_size,
            "gpu_peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
        }
    )
    (output / "mining_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
