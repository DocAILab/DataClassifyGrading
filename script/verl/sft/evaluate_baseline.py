"""SFT baseline evaluator: generate with a model and score choice-protocol output.

Runs a model (base HF model or a LoRA-merged HF directory) over a Phase-8 SFT
parquet split and reports the two-stage choice-protocol metrics using the SHARED
choice-aware evaluation layer (``agent.evaluation.classification``), so the
metrics exactly match the frozen parser / contract (no re-implementation).

For each row the prompt is the exported conversation WITHOUT the assistant gold
(loss target is never shown at inference). Generation is fixed and identical
for every model: greedy, ``do_sample=False``, ``num_beams=1``, fixed
``max_new_tokens`` and seed.

Metrics (per the Phase-13 spec):
- stage1: format_valid, contract_valid, recall@5 (GT in the 5 predicted ids)
- stage2: format_valid, contract_valid, accuracy-overall,
          accuracy-when-GT-in-candidates (and counts)
- end-to-end: stage1 recall (same source) AND stage2 correct, per source_id
  (stage1 recall x stage2 final correctness)

Usage (server, in the SFT venv):
  python -m script.verl.sft.evaluate_baseline \
    --model-path <hf dir or merged dir> \
    --data data/sft/pers_info/test.parquet \
    --registry cfg/task/registry/pers_info.registry.json \
    --max-new-tokens 128 --seed 42 --report <out.json>
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable
import sys

from agent.evaluation.classification import (
    evaluate_stage1_choices,
    evaluate_stage2_choices,
)
from agent.task.contracts import LeafRegistry
from agent.task.prompt_choices import PromptChoiceRegistry

FULL_CATALOG_REGISTRY = "pers_info"  # evaluator is registry-driven, not hard-coded


def aggregate_baseline(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-row evaluation records into the Phase-13 metric table.

    ``records`` items: {stage, source_id, ground_truth, candidates,
    format_valid, contract_valid, correct, recalled} (choice-decode already
    applied by the shared evaluator). Pure function, no transformers/torch.
    """
    by_stage: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_stage[record["stage"]].append(record)

    def summarize(stage_rows: Iterable[dict]) -> dict[str, Any]:
        rows = list(stage_rows)
        n = len(rows)
        format_valid = sum(r["format_valid"] for r in rows) / n if n else 0.0
        contract_valid = sum(r["contract_valid"] for r in rows) / n if n else 0.0
        return {
            "n": n,
            "format_valid": format_valid,
            "contract_valid": contract_valid,
        }

    # --- stage 1 ---
    s1 = dict(summarize(by_stage["stage1"]))
    n_s1 = s1["n"]
    s1["recall_at_5"] = sum(r["recalled"] for r in by_stage["stage1"]) / n_s1 if n_s1 else 0.0
    s1["recalled_count"] = sum(r["recalled"] for r in by_stage["stage1"])

    # --- stage 2 ---
    s2 = dict(summarize(by_stage["stage2"]))
    n_s2 = s2["n"]
    correct = sum(r["correct"] for r in by_stage["stage2"])
    s2["accuracy_overall"] = correct / n_s2 if n_s2 else 0.0
    gt_in = [r for r in by_stage["stage2"] if r["gt_in_candidates"]]
    s2["n_gt_in_candidates"] = len(gt_in)
    s2["accuracy_when_gt_in_candidates"] = (
        sum(r["correct"] for r in gt_in) / len(gt_in) if gt_in else 0.0
    )
    s2["correct_count"] = correct

    # --- end-to-end: stage1 recall AND stage2 correct on the same source ---
    by_source: dict[str, dict[str, bool]] = defaultdict(dict)
    for r in records:
        by_source[r["source_id"]][r["stage"]] = r
    e2e_correct = sum(
        1
        for per_source in by_source.values()
        if per_source.get("stage1", {}).get("recalled")
        and per_source.get("stage2", {}).get("correct")
    )
    e2e = {
        "pairs": len(by_source),
        "correct": e2e_correct,
        "correct_rate": e2e_correct / len(by_source) if by_source else 0.0,
    }

    return {"stage1": s1, "stage2": s2, "end_to_end": e2e}


def _evaluate_rows(
    rows: list[dict[str, Any]],
    registry: LeafRegistry,
) -> list[dict[str, Any]]:
    choices = PromptChoiceRegistry.from_registry(registry)
    records: list[dict[str, Any]] = []
    for row in rows:
        stage = row["stage"]
        gt = row["ground_truth"]
        source_id = row["source_id"]
        if stage == "stage1":
            evaluation = evaluate_stage1_choices(
                row["completion"], ground_truth=gt, registry=registry, choices=choices
            )
            records.append(
                {
                    "stage": "stage1",
                    "source_id": source_id,
                    "ground_truth": gt,
                    "candidates": None,
                    "format_valid": evaluation.format_valid,
                    "contract_valid": evaluation.contract_valid,
                    "recalled": evaluation.ground_truth_recalled,
                    "correct": False,
                    "gt_in_candidates": False,
                    "prediction": evaluation.prediction,
                }
            )
        else:
            candidates = list(row["candidates"])
            evaluation = evaluate_stage2_choices(
                row["completion"], ground_truth=gt, candidates=candidates, registry=registry
            )
            records.append(
                {
                    "stage": "stage2",
                    "source_id": source_id,
                    "ground_truth": gt,
                    "candidates": candidates,
                    "format_valid": evaluation.format_valid,
                    "contract_valid": evaluation.contract_valid,
                    "correct": evaluation.correct,
                    "gt_in_candidates": gt in candidates,
                    "recalled": False,
                    "prediction": evaluation.prediction,
                }
            )
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="HF model dir (base or LoRA-merged)")
    parser.add_argument("--data", required=True, help="SFT parquet split (e.g. test.parquet)")
    parser.add_argument("--registry", required=True, help="Leaf registry JSON")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report", required=True, help="metrics JSON output path")
    parser.add_argument("--completions", help="optional path to precomputed completions JSONL; skips generation")
    args = parser.parse_args(argv)

    import pyarrow.parquet as pq

    registry = LeafRegistry.from_path(args.registry)
    rows = pq.read_table(args.data).to_pylist()

    generation: dict[str, Any] = {
        "model_path": args.model_path,
        "do_sample": False,
        "num_beams": 1,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
    }

    if args.completions:
        records = []
        # completions JSONL: one {"source_id"/"stage"/"completion"} per eval row
        with open(args.completions, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    records.append(json.loads(line))
        # rows are in parquet order -> match completion entries by identity (stage+source_id)
        by_key = {(r["stage"], r["source_id"]): r for r in records}
        enriched = []
        for row in rows:
            key = (row["stage"], row["source_id"])
            if key not in by_key:
                print(f"error: missing completion for {key}", file=sys.stderr)
                return 2
            enriched.append({**row, "completion": by_key[key]["completion"]})
        rows = enriched
    else:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.manual_seed(args.seed)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16, device_map="auto"
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        model.eval()
        import tqdm

        for index in tqdm.tqdm(range(len(rows)), desc="generate"):
            row = rows[index]
            messages = row["messages"][:2]  # system + user, no assistant gold
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            with torch.inference_mode():
                output = model.generate(
                    **inputs,
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.eos_token_id,
                )
            input_len = inputs["input_ids"].shape[-1]
            rows[index]["completion"] = tokenizer.decode(
                output[0][input_len:], skip_special_tokens=True
            ).strip()

    records = _evaluate_rows(rows, registry)
    metrics = aggregate_baseline(records)

    report = {
        "metrics": metrics,
        "generation": generation,
        "per_row": records,
        "registry": str(args.registry),
        "data": str(args.data),
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    path = Path(args.report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
