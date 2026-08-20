"""Generate and evaluate the official field-name-only two-stage SFT baseline."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from agent.evaluation import evaluate_stage1, evaluate_stage2
from agent.task.contracts import LeafRegistry, TaskConfig
from agent.task.prompts import build_stage1_prompt, build_stage2_prompt


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def summarize_predictions(
    rows: Sequence[Mapping[str, Any]], *, registry_ids: Sequence[str]
) -> dict[str, Any]:
    source_ids = [str(row["source_id"]) for row in rows]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("validation predictions must have unique source_id values")
    registry = set(registry_ids)
    if not rows:
        raise ValueError("validation predictions must not be empty")
    if any(str(row["golden_level_4"]) not in registry for row in rows):
        raise ValueError("every golden_level_4 must belong to the registry")

    total = len(rows)
    stage1_valid = sum(bool(row["stage1_contract_valid"]) for row in rows)
    recalled = sum(bool(row["stage1_recalled"]) for row in rows)
    attempted = sum(bool(row["stage2_attempted"]) for row in rows)
    stage2_valid = sum(bool(row["stage2_contract_valid"]) for row in rows)
    conditional_rows = [row for row in rows if bool(row["stage1_recalled"])]
    conditional_correct = sum(
        row.get("prediction") == row["golden_level_4"] for row in conditional_rows
    )
    end_to_end_correct = sum(
        row.get("prediction") == row["golden_level_4"] for row in rows
    )

    f1_values: list[float] = []
    for label in sorted({str(row["golden_level_4"]) for row in rows}):
        true_positive = sum(
            row["golden_level_4"] == label and row.get("prediction") == label
            for row in rows
        )
        false_positive = sum(
            row["golden_level_4"] != label and row.get("prediction") == label
            for row in rows
        )
        false_negative = sum(
            row["golden_level_4"] == label and row.get("prediction") != label
            for row in rows
        )
        denominator = 2 * true_positive + false_positive + false_negative
        f1_values.append(_safe_ratio(2 * true_positive, denominator))

    return {
        "examples": total,
        "stage1_contract_valid": stage1_valid,
        "stage1_contract_valid_rate": _safe_ratio(stage1_valid, total),
        "stage1_recalled": recalled,
        "stage1_recall_at_5": _safe_ratio(recalled, total),
        "stage2_attempted": attempted,
        "stage2_contract_valid": stage2_valid,
        "stage2_contract_valid_rate_among_attempted": _safe_ratio(stage2_valid, attempted),
        "stage2_conditional_denominator": len(conditional_rows),
        "stage2_conditional_correct": conditional_correct,
        "stage2_conditional_accuracy": _safe_ratio(
            conditional_correct, len(conditional_rows)
        ),
        "end_to_end_correct": end_to_end_correct,
        "end_to_end_accuracy": _safe_ratio(end_to_end_correct, total),
        "macro_f1": sum(f1_values) / len(f1_values),
        "gold_labels": len(f1_values),
        "invalid_or_oov_predictions": sum(
            row.get("prediction") not in registry for row in rows
        ),
    }


def _load_validation(path: Path, limit: int | None) -> list[dict[str, Any]]:
    if path.name != "val.json":
        raise ValueError("only an explicit val.json split is permitted")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError("val.json must contain a JSON array")
    records = value if limit is None else value[:limit]
    if not records or not all(isinstance(item, dict) for item in records):
        raise ValueError("val.json must contain record objects")
    return records


def _record_fields(item: Mapping[str, Any]) -> tuple[str, str, str]:
    source_id = str(item.get("id", "")).strip()
    metadata = item.get("metadata")
    classification = item.get("classification")
    if not source_id or not isinstance(metadata, Mapping) or not isinstance(classification, Mapping):
        raise ValueError("invalid validation record")
    field_name = "" if metadata.get("field_name") is None else str(metadata.get("field_name", ""))
    golden = str(classification.get("level_4", "")).strip()
    if not golden:
        raise ValueError(f"validation record {source_id!r} has no level_4")
    return source_id, field_name, golden


def _batches(items: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _generate(model, tokenizer, messages: Sequence[Sequence[dict[str, str]]], max_new_tokens: int) -> list[str]:
    import torch

    rendered = [
        tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        for message in messages
    ]
    inputs = tokenizer(rendered, return_tensors="pt", padding=True)
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    input_width = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.batch_decode(generated[:, input_width:], skip_special_tokens=True)


def _append_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _read_jsonl_by_id(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    result: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            result[str(row["source_id"])] = row
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for SFT evaluation")
    records = _load_validation(args.val, args.limit)
    registry = LeafRegistry.from_path(args.registry)
    config = TaskConfig(("field_name",))
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    stage1_path = output / "stage1_predictions.jsonl"
    final_path = output / "predictions.jsonl"

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        local_files_only=True,
        attn_implementation="sdpa",
        device_map={"": 0},
    )
    model = PeftModel.from_pretrained(
        base, args.adapter, local_files_only=True, is_trainable=False
    )
    model.eval()
    started = time.perf_counter()

    stage1 = _read_jsonl_by_id(stage1_path)
    pending = [item for item in records if _record_fields(item)[0] not in stage1]
    for batch in _batches(pending, args.batch_size):
        prepared = []
        messages = []
        for item in batch:
            source_id, field_name, golden = _record_fields(item)
            prompt = build_stage1_prompt({"field_name": field_name}, registry, config)
            prepared.append((source_id, field_name, golden, prompt))
            messages.append([
                {"role": "system", "content": prompt.system},
                {"role": "user", "content": prompt.user},
            ])
        raw_outputs = _generate(model, tokenizer, messages, args.stage1_max_new_tokens)
        emitted = []
        for (source_id, field_name, golden, prompt), raw in zip(prepared, raw_outputs):
            evaluation = evaluate_stage1(raw, ground_truth=golden, registry=registry)
            emitted.append({
                "source_id": source_id,
                "field_name": field_name,
                "golden_level_4": golden,
                "prompt": {"system": prompt.system, "user": prompt.user},
                "raw_output": raw,
                "candidates": list(evaluation.prediction) if evaluation.prediction else None,
                "stage1_format_valid": evaluation.format_valid,
                "stage1_contract_valid": evaluation.contract_valid,
                "stage1_recalled": evaluation.ground_truth_recalled,
                "stage1_errors": list(evaluation.errors),
            })
        _append_jsonl(stage1_path, emitted)
        stage1.update({row["source_id"]: row for row in emitted})

    completed = _read_jsonl_by_id(final_path)
    pending_ids = [
        _record_fields(item)[0]
        for item in records
        if _record_fields(item)[0] not in completed
    ]
    for id_batch in _batches(pending_ids, args.batch_size):
        prepared = []
        messages = []
        emitted = []
        for source_id in id_batch:
            first = stage1[source_id]
            candidates = first["candidates"]
            if not first["stage1_contract_valid"] or not isinstance(candidates, list):
                emitted.append({
                    **first,
                    "stage2_attempted": False,
                    "stage2_raw_output": None,
                    "stage2_format_valid": False,
                    "stage2_contract_valid": False,
                    "stage2_errors": ["skipped because Stage 1 contract was invalid"],
                    "prediction": None,
                    "correct": False,
                })
                continue
            prompt = build_stage2_prompt(
                {"field_name": first["field_name"]}, candidates, registry, config
            )
            prepared.append((first, prompt))
            messages.append([
                {"role": "system", "content": prompt.system},
                {"role": "user", "content": prompt.user},
            ])
        if prepared:
            raw_outputs = _generate(model, tokenizer, messages, args.stage2_max_new_tokens)
            for (first, prompt), raw in zip(prepared, raw_outputs):
                evaluation = evaluate_stage2(
                    raw,
                    ground_truth=first["golden_level_4"],
                    candidates=first["candidates"],
                    registry=registry,
                )
                emitted.append({
                    **first,
                    "stage2_prompt": {"system": prompt.system, "user": prompt.user},
                    "stage2_attempted": True,
                    "stage2_raw_output": raw,
                    "stage2_format_valid": evaluation.format_valid,
                    "stage2_contract_valid": evaluation.contract_valid,
                    "stage2_errors": list(evaluation.errors),
                    "prediction": evaluation.prediction,
                    "correct": evaluation.correct,
                })
        _append_jsonl(final_path, emitted)
        completed.update({row["source_id"]: row for row in emitted})

    ordered = [completed[_record_fields(item)[0]] for item in records]
    metrics = summarize_predictions(ordered, registry_ids=registry.ids)
    report = {
        "model": str(args.model),
        "adapter": str(args.adapter),
        "split": "val",
        "requested_splits": ["val"],
        "real_test_split_read": False,
        "metadata_fields": ["field_name"],
        "supervision_target": "classification.level_4",
        "decoding": {"do_sample": False, "batch_size": args.batch_size},
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "metrics": metrics,
        "predictions": str(final_path),
    }
    (output / "evaluation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "COMPLETE").write_text("complete\n", encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--stage1-max-new-tokens", type=int, default=128)
    parser.add_argument("--stage2-max-new-tokens", type=int, default=64)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    print(json.dumps(run(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
