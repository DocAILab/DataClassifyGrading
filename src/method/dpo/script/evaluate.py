"""Evaluate SFT and DPO on identical val-only semantic hard candidates."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time
from typing import Sequence

from agent.task import LeafRegistry, TaskConfig
from method.dpo.evaluation import (
    build_evaluation_case,
    evaluate_cases,
    paired_classification_report,
)
from method.dpo.label_scoring import score_candidate_answers
from method.dpo.training import load_merged_sft_model, validate_runtime_versions
from method.sft.dataset import load_json_records


def load_val_records(input_dir: str | Path) -> list[dict]:
    """Load val.json directly without resolving or opening test.json."""
    return load_json_records(Path(input_dir) / "val.json")


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--task-config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--sft-adapter", required=True)
    parser.add_argument("--dpo-adapter", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=137)
    return parser


def _evaluate_policy(
    *,
    model_path: str,
    sft_adapter: str,
    dpo_adapter: str | None,
    cases: list[dict],
    output_path: Path,
    batch_size: int,
) -> dict:
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_merged_sft_model(model_path, sft_adapter)
    if dpo_adapter is not None:
        model = PeftModel.from_pretrained(model, dpo_adapter, is_trainable=False)
    model.eval()

    def score_fn(prompt, answers):
        return score_candidate_answers(
            model,
            tokenizer,
            prompt,
            answers,
            batch_size=batch_size,
            device="cuda",
        )

    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    report = evaluate_cases(cases, output_path, score_fn=score_fn)
    report["elapsed_seconds"] = time.monotonic() - started
    report["gpu_peak_memory_bytes"] = int(torch.cuda.max_memory_allocated())
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    versions = validate_runtime_versions()
    registry = LeafRegistry.from_path(args.registry)
    config = TaskConfig.from_path(args.task_config)
    if config.metadata_fields != ("field_name",):
        raise ValueError("evaluation requires metadata_fields=[field_name]")
    records = load_val_records(args.input_dir)
    cases = [build_evaluation_case(record, registry, seed=args.seed) for record in records]
    source_ids = [case["source_id"] for case in cases]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("validation source_id values must be unique")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    audit = {
        "requested_splits": ["val"],
        "real_test_split_read": False,
        "metadata_fields": ["field_name"],
        "candidate_policy": "semantic_hard_oracle_shuffled_v1",
        "candidate_seed": args.seed,
        "rows": len(cases),
        "unique_source_ids": len(set(source_ids)),
        "runtime_versions": versions,
    }
    (output / "data_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    sft_path = output / "sft_predictions.jsonl"
    dpo_path = output / "dpo_predictions.jsonl"
    sft_runtime = _evaluate_policy(
        model_path=args.model,
        sft_adapter=args.sft_adapter,
        dpo_adapter=None,
        cases=cases,
        output_path=sft_path,
        batch_size=args.batch_size,
    )
    dpo_runtime = _evaluate_policy(
        model_path=args.model,
        sft_adapter=args.sft_adapter,
        dpo_adapter=args.dpo_adapter,
        cases=cases,
        output_path=dpo_path,
        batch_size=args.batch_size,
    )
    comparison = paired_classification_report(
        _read_jsonl(sft_path), _read_jsonl(dpo_path), registry.ids
    )
    comparison.update(
        {
            "requested_splits": ["val"],
            "real_test_split_read": False,
            "metadata_fields": ["field_name"],
            "sft_runtime": sft_runtime,
            "dpo_runtime": dpo_runtime,
        }
    )
    (output / "comparison_to_sft.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "COMPLETE").write_text("complete\n", encoding="utf-8")
    print(json.dumps(comparison, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
