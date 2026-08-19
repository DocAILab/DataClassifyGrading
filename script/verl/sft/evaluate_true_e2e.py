"""TRUE end-to-end evaluator for the two-stage choice-protocol task.

Unlike the factorized/proxy evaluator (``evaluate_baseline.py`` — Stage 2 is
scored against the pre-constructed gold-containing bundle from the parquet),
this evaluator chains the REAL task pipeline on a raw test sample:

    raw test sample (stage1)
      -> build Stage1 prompt (reuses agent.task.prompts.build_stage1_prompt)
      -> model greedy generate
      -> shared choice parser decode -> REAL canonical top-5
      -> fetch description/descriptions/examples for those 5 category ids
         from the canonical corpus (reuses build_stage2_prompt)
      -> dynamic Stage2 prompt from the PREDICTED top-5
      -> model greedy generate
      -> local bundle-id decode -> final canonical category_id
      -> compare against ground truth

Every identity/decode/prompt step is the FROZEN shared layer
(agent.task.prompts / PromptChoiceRegistry / check_stage{1,2}_choices /
LeafRegistry + canonical corpus). Nothing is copied, no fuzzy fallback.

Metrics:
- stage1: format-valid rate, contract-valid rate, Recall@5
- stage2: conditional accuracy over sources with GT in the PREDICTED top-5,
          format/contract failure rates (over attempted stage2)
- true end-to-end accuracy (stage1 recall AND stage2 correct, per source)

Generation is injected (``generate(messages) -> str``) so unit tests drive the
full pipeline on CPU with canned model output; the CLI wraps a fixed greedy
transformers decoder (identical decoding settings for base vs SFT).

Usage (server, SFT venv):
  python -m script.verl.sft.evaluate_true_e2e \
    --model-path <hf dir or merged dir> \
    --data <phase-8 test.parquet> \
    --registry cfg/task/registry/pers_info.registry.json \
    --corpus cfg/task/corpus/pers_info.corpus.json \
    --metadata-fields field_name field_description \
    --max-new-tokens 128 --seed 42 --report <out.json>
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping
import sys

from agent.evaluation.classification import (
    evaluate_stage1_choices,
    evaluate_stage2_choices,
)
from agent.task.contracts import CorpusCategory, LeafRegistry, TaskConfig
from agent.task.prompts import (
    Prompt,
    build_stage1_prompt,
    build_stage2_prompt,
)

Split = tuple[str, str, str]  # (system, user, -) placeholder for typing simplicity

GenerateFn = Callable[[list[dict[str, str]]], str]


@dataclass(frozen=True)
class E2EOutcome:
    """One source's TRUE end-to-end result (choice decode already applied)."""

    source_id: str
    ground_truth: str
    # stage 1
    stage1_format_valid: bool
    stage1_contract_valid: bool
    recalled: bool  # GT in the predicted top-5
    predicted_top5: tuple[str, ...] | None
    stage1_completion: str
    # stage 2 (None when stage1 was contract-invalid -> never prompted)
    stage2_attempted: bool
    stage2_prompt_candidates: tuple[str, ...] | None  # == predicted top-5
    stage2_format_valid: bool
    stage2_contract_valid: bool
    stage2_correct: bool
    final_decision: str | None
    stage2_completion: str | None
    failures: tuple[str, ...] = field(default_factory=tuple)

    @property
    def e2e_correct(self) -> bool:
        return self.recalled and self.stage2_correct


def run_one(
    source: dict[str, Any],
    *,
    registry: LeafRegistry,
    config: TaskConfig,
    corpus: Mapping[str, CorpusCategory] | None,
    seed: int,
    generate: GenerateFn,
) -> E2EOutcome:
    """Run the true pipeline for one source (a stage1 row from the parquet)."""
    metadata = source["metadata"]
    ground_truth = source["ground_truth"]
    stage1_prompt = build_stage1_prompt(metadata, registry, config)
    stage1_completion = generate(
        [
            {"role": "system", "content": stage1_prompt.system},
            {"role": "user", "content": stage1_prompt.user},
        ]
    )
    stage1_eval = evaluate_stage1_choices(
        stage1_completion, ground_truth=ground_truth, registry=registry
    )

    base: dict[str, Any] = {
        "source_id": source["source_id"],
        "ground_truth": ground_truth,
        "stage1_format_valid": stage1_eval.format_valid,
        "stage1_contract_valid": stage1_eval.contract_valid,
        "recalled": stage1_eval.ground_truth_recalled and stage1_eval.contract_valid,
        "predicted_top5": stage1_eval.prediction,
        "stage1_completion": stage1_completion,
        "stage2_attempted": False,
        "stage2_prompt_candidates": None,
        "stage2_format_valid": False,
        "stage2_contract_valid": False,
        "stage2_correct": False,
        "final_decision": None,
        "stage2_completion": None,
        "failures": stage1_eval.errors,
    }
    if not stage1_eval.contract_valid:
        return E2EOutcome(**base)

    predicted = tuple(stage1_eval.prediction)  # non-None because contract-valid
    try:
        stage2_prompt: Prompt = build_stage2_prompt(
            metadata, predicted, registry, config, corpus=corpus
        )
    except ValueError as exc:  # e.g. predicted id absent from canonical corpus
        base["failures"] = base["failures"] + (f"stage2 build: {exc}",)
        return E2EOutcome(**base)

    stage2_completion = generate(
        [
            {"role": "system", "content": stage2_prompt.system},
            {"role": "user", "content": stage2_prompt.user},
        ]
    )
    stage2_eval = evaluate_stage2_choices(
        stage2_completion, ground_truth=ground_truth, candidates=predicted, registry=registry
    )
    base.update(
        {
            "stage2_attempted": True,
            "stage2_prompt_candidates": predicted,
            "stage2_format_valid": stage2_eval.format_valid,
            "stage2_contract_valid": stage2_eval.contract_valid,
            "stage2_correct": stage2_eval.correct,
            "final_decision": stage2_eval.prediction,
            "stage2_completion": stage2_completion,
            "failures": stage2_eval.errors,
        }
    )
    return E2EOutcome(**base)


def aggregate_true_e2e(outcomes: list[E2EOutcome]) -> dict[str, Any]:
    """Aggregate per-source E2E outcomes into the published metric table."""
    n = len(outcomes)
    stage1_fmt = sum(o.stage1_format_valid for o in outcomes) / n if n else 0.0
    stage1_contract = sum(o.stage1_contract_valid for o in outcomes) / n if n else 0.0
    recalled = sum(o.recalled for o in outcomes)
    attempted = sum(o.stage2_attempted for o in outcomes)
    s2_fmt_fail = (
        sum(1 for o in outcomes if o.stage2_attempted and not o.stage2_format_valid) / attempted
        if attempted
        else 0.0
    )
    s2_contract_fail = (
        sum(1 for o in outcomes if o.stage2_attempted and not o.stage2_contract_valid) / attempted
        if attempted
        else 0.0
    )
    s2_correct = sum(o.stage2_correct for o in outcomes)
    conditional_acc = s2_correct / recalled if recalled else 0.0
    e2e_correct = sum(o.e2e_correct for o in outcomes)
    return {
        "sources": n,
        "stage1_format_valid": stage1_fmt,
        "stage1_contract_valid": stage1_contract,
        "stage1_recall_at_5": recalled / n if n else 0.0,
        "stage1_recalled_count": recalled,
        "stage2_attempted": attempted,
        "stage2_format_failure_rate": s2_fmt_fail,
        "stage2_contract_failure_rate": s2_contract_fail,
        "stage2_conditional_accuracy": conditional_acc,
        "true_e2e_accuracy": e2e_correct / n if n else 0.0,
        "true_e2e_correct": e2e_correct,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="HF model dir (base or merged)")
    parser.add_argument("--data", required=True, help="phase-8 SFT parquet (test split)")
    parser.add_argument("--registry", required=True, help="leaf registry JSON")
    parser.add_argument("--corpus", required=True, help="canonical corpus JSON")
    parser.add_argument("--metadata-fields", nargs="+", default=["field_name", "field_description"])
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report", required=True)
    args = parser.parse_args(argv)

    import pyarrow.parquet as pq

    registry = LeafRegistry.from_path(args.registry)
    config = TaskConfig(metadata_fields=tuple(args.metadata_fields))
    from agent.task.canonical_dataset import load_corpus_categories

    corpus = {
        category.category_id: category
        for category in load_corpus_categories(args.corpus)
    }
    rows = pq.read_table(args.data).to_pylist()
    stage1_rows = [r for r in rows if r["stage"] == "stage1"]

    # greedy, fixed decoding settings (identical for base and SFT), model loaded once
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

    def hf_generate(messages: list[dict[str, str]]) -> str:
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
        return tokenizer.decode(
            output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
        ).strip()

    outcomes = [
        run_one(row, registry=registry, config=config, corpus=corpus,
                seed=args.seed, generate=hf_generate)
        for row in stage1_rows
    ]
    metrics = aggregate_true_e2e(outcomes)
    report = {
        "metrics": metrics,
        "generation": {
            "model_path": args.model_path,
            "do_sample": False,
            "num_beams": 1,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
        },
        "per_source": [
            {
                "source_id": o.source_id,
                "ground_truth": o.ground_truth,
                "stage1_format_valid": o.stage1_format_valid,
                "stage1_contract_valid": o.stage1_contract_valid,
                "recalled": o.recalled,
                "predicted_top5": o.predicted_top5,
                "stage1_completion": o.stage1_completion,
                "stage2_attempted": o.stage2_attempted,
                "stage2_prompt_candidates": o.stage2_prompt_candidates,
                "stage2_format_valid": o.stage2_format_valid,
                "stage2_contract_valid": o.stage2_contract_valid,
                "stage2_correct": o.stage2_correct,
                "final_decision": o.final_decision,
                "stage2_completion": o.stage2_completion,
                "e2e_correct": o.e2e_correct,
                "failures": list(o.failures),
            }
            for o in outcomes
        ],
        "registry": str(args.registry),
        "corpus": str(args.corpus),
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
