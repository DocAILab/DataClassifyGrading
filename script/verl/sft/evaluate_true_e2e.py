"""End-to-end evaluator for the two-stage choice-protocol task.

The evaluator builds a Stage 1 prompt, decodes the predicted canonical top-5,
builds Stage 2 from exactly those candidates, and scores the final canonical
category id. Generation is injected for CPU tests; the CLI supplies a fixed
greedy transformers adapter.

All registry, corpus, task, data, model, and report locations are explicit
runtime-local paths. No production asset path is embedded in the repository.

Example:
  python -m script.verl.sft.evaluate_true_e2e \
    --model-path <local-model-dir> --data <local-test.parquet> \
    --registry <local-registry.json> --corpus <local-corpus.json> \
    --task-config <local-task.json> --report <local-report.json>
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
from agent.task.assets import ClassificationAssets
from agent.task.contracts import (
    CorpusCategory,
    GradedTaskContext,
    GradingConfig,
    LeafRegistry,
    TaskConfig,
)
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
    # Joint grading diagnostics.  These remain optional for legacy
    # classification-only runs so existing reports retain their shape.
    ground_truth_level: str | None = None
    predicted_level: str | None = None
    leaf_correct: bool = False
    level_correct: bool = False

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
    grading: GradingConfig | None = None,
) -> E2EOutcome:
    """Run the true pipeline for one source (a stage1 row from the parquet).

    In joint mode every source must carry the exported ``ground_truth_level``;
    omitting it is a data-contract error rather than permission to fall back to
    classification-only scoring.
    """
    metadata = source["metadata"]
    ground_truth = source["ground_truth"]
    if grading is None:
        graded = GradedTaskContext()
        ground_truth_level = None
    else:
        raw_level = source.get("ground_truth_level")
        if not isinstance(raw_level, str) or not raw_level.strip():
            raise ValueError(
                "joint true-E2E source requires a non-empty ground_truth_level"
            )
        ground_truth_level = raw_level.strip()
        graded = GradedTaskContext(grading, ground_truth_level)
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
        "ground_truth_level": ground_truth_level,
        "predicted_level": None,
        "leaf_correct": False,
        "level_correct": False,
    }
    if not stage1_eval.contract_valid:
        return E2EOutcome(**base)

    predicted = tuple(stage1_eval.prediction)  # non-None because contract-valid
    try:
        stage2_prompt: Prompt = build_stage2_prompt(
            metadata,
            predicted,
            registry,
            config,
            corpus=corpus,
            grading=graded.grading,
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
        stage2_completion,
        ground_truth=ground_truth,
        candidates=predicted,
        registry=registry,
        grading=graded.grading,
        expected_level=graded.expected_level,
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
            "predicted_level": stage2_eval.predicted_level,
            "leaf_correct": (
                stage2_eval.contract_valid
                and stage2_eval.prediction == ground_truth
            ),
            "level_correct": stage2_eval.level_correct,
        }
    )
    return E2EOutcome(**base)


def _macro_f1(
    truths: list[str], predictions: list[str | None]
) -> float:
    """Compute macro-F1 without a third-party metrics dependency.

    Invalid/missing predictions remain false negatives for their true class;
    predicted labels not present in the truth set still contribute false
    positives. This keeps format failures visible instead of dropping rows.
    """
    if not truths:
        return 0.0
    # The evaluation label universe is fixed by ground-truth support.  A
    # prediction outside that universe still creates a false negative for its
    # source's true class, but must not change the macro denominator from run
    # to run.
    labels = set(truths)
    scores: list[float] = []
    for label in labels:
        true_positive = sum(
            truth == label and prediction == label
            for truth, prediction in zip(truths, predictions)
        )
        false_positive = sum(
            truth != label and prediction == label
            for truth, prediction in zip(truths, predictions)
        )
        false_negative = sum(
            truth == label and prediction != label
            for truth, prediction in zip(truths, predictions)
        )
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(
            2 * true_positive / denominator if denominator else 0.0
        )
    return sum(scores) / len(scores) if scores else 0.0


def aggregate_true_e2e(outcomes: list[E2EOutcome]) -> dict[str, Any]:
    """Aggregate strict joint EM and diagnostic per-head metrics.

    ``true_e2e_accuracy`` remains the legacy name for strict leaf E2E EM.
    When joint grading is enabled, ``strict_joint_em`` requires both the
    canonical leaf and ``data_level`` to match. ``composite_macro_f1`` treats
    the pair ``(canonical leaf, data_level)`` as one composite class; it is
    deliberately not the average of the two per-head F1 diagnostics.
    """
    n = len(outcomes)
    stage1_fmt = sum(o.stage1_format_valid for o in outcomes) / n if n else 0.0
    stage1_contract = sum(o.stage1_contract_valid for o in outcomes) / n if n else 0.0
    recalled = sum(o.recalled for o in outcomes)
    attempted = sum(o.stage2_attempted for o in outcomes)
    valid_formats = sum(
        o.stage2_attempted and o.stage2_format_valid for o in outcomes
    )
    valid_contracts = sum(
        o.stage2_attempted and o.stage2_contract_valid for o in outcomes
    )
    s2_fmt_fail = (attempted - valid_formats) / attempted if attempted else 0.0
    s2_contract_fail = (attempted - valid_contracts) / attempted if attempted else 0.0
    s2_correct = sum(o.stage2_correct for o in outcomes)
    conditional_acc = s2_correct / recalled if recalled else 0.0
    e2e_correct = sum(o.e2e_correct for o in outcomes)

    leaf_truths = [o.ground_truth for o in outcomes]
    leaf_predictions = [
        o.final_decision if o.stage2_contract_valid else None for o in outcomes
    ]
    leaf_correct = sum(
        prediction is not None and prediction == truth
        for truth, prediction in zip(leaf_truths, leaf_predictions)
    )
    leaf_macro_f1 = _macro_f1(leaf_truths, leaf_predictions)

    joint_enabled = any(o.ground_truth_level is not None for o in outcomes)
    if joint_enabled and any(o.ground_truth_level is None for o in outcomes):
        raise ValueError("cannot aggregate mixed joint and classification-only outcomes")
    level_truths = [
        o.ground_truth_level for o in outcomes if o.ground_truth_level is not None
    ]
    level_predictions = [
        o.predicted_level if o.ground_truth_level is not None and o.stage2_contract_valid else None
        for o in outcomes
        if o.ground_truth_level is not None
    ]
    level_correct = sum(
        prediction is not None and prediction == truth
        for truth, prediction in zip(level_truths, level_predictions)
    )
    level_macro_f1 = _macro_f1(level_truths, level_predictions)
    strict_joint_correct = sum(
        o.e2e_correct and (not joint_enabled or o.level_correct) for o in outcomes
    )
    strict_joint_em = strict_joint_correct / n if n else 0.0
    leaf_em = leaf_correct / n if n else 0.0
    level_em = level_correct / len(level_truths) if level_truths else 0.0
    if joint_enabled:
        composite_truths = [
            json.dumps(
                [outcome.ground_truth, outcome.ground_truth_level],
                ensure_ascii=False,
                separators=(",", ":"),
            )
            for outcome in outcomes
        ]
        composite_predictions = [
            json.dumps(
                [outcome.final_decision, outcome.predicted_level],
                ensure_ascii=False,
                separators=(",", ":"),
            )
            if outcome.stage2_contract_valid
            and outcome.final_decision is not None
            and outcome.predicted_level is not None
            else None
            for outcome in outcomes
        ]
        composite_macro_f1 = _macro_f1(
            composite_truths, composite_predictions
        )
    else:
        composite_macro_f1 = leaf_macro_f1

    leaf_head = {
        "support": n,
        "exact_match": leaf_em,
        "correct": leaf_correct,
        "macro_f1": leaf_macro_f1,
        "format_rate": valid_formats / attempted if attempted else 0.0,
    }
    level_head = {
        "support": len(level_truths),
        "exact_match": level_em,
        "correct": level_correct,
        "macro_f1": level_macro_f1,
        "format_rate": valid_formats / attempted if attempted else 0.0,
    }
    return {
        "sources": n,
        "stage1_format_valid": stage1_fmt,
        "stage1_contract_valid": stage1_contract,
        "stage1_recall_at_5": recalled / n if n else 0.0,
        "stage1_recalled_count": recalled,
        "stage2_attempted": attempted,
        "stage2_format_rate": valid_formats / attempted if attempted else 0.0,
        "stage2_format_rate_all": valid_formats / n if n else 0.0,
        "stage2_contract_rate": valid_contracts / attempted if attempted else 0.0,
        "stage2_format_failure_rate": s2_fmt_fail,
        "stage2_contract_failure_rate": s2_contract_fail,
        "stage2_conditional_accuracy": conditional_acc,
        "leaf_exact_match": leaf_em,
        "leaf_macro_f1": leaf_macro_f1,
        "category_exact_match": leaf_em,
        "category_macro_f1": leaf_macro_f1,
        "data_level_exact_match": level_em,
        "data_level_macro_f1": level_macro_f1,
        "strict_joint_em": strict_joint_em,
        "strict_joint_em_count": strict_joint_correct,
        # Short aliases make the report convenient for downstream dashboards.
        "joint_em": strict_joint_em,
        "joint_exact_match": strict_joint_em,
        "composite_macro_f1": composite_macro_f1,
        "per_head": {
            "leaf": leaf_head,
            "category": leaf_head,
            "data_level": level_head,
        },
        "true_e2e_accuracy": e2e_correct / n if n else 0.0,
        "true_e2e_correct": e2e_correct,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="HF model dir (base or merged)")
    parser.add_argument("--data", required=True, help="phase-8 SFT parquet (test split)")
    parser.add_argument("--registry", required=True, help="leaf registry JSON")
    parser.add_argument("--corpus", required=True, help="canonical corpus JSON")
    parser.add_argument(
        "--grading-config",
        default=None,
        help=(
            "Optional grading JSON shared with SFT export; when supplied, "
            "Stage 2 must emit both answer and level"
        ),
    )
    task_group = parser.add_mutually_exclusive_group(required=True)
    task_group.add_argument("--task-config", help="local task configuration JSON")
    task_group.add_argument(
        "--metadata-fields",
        nargs="+",
        help="explicit metadata fields (alternative to --task-config)",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report", required=True)
    args = parser.parse_args(argv)

    import pyarrow.parquet as pq

    task = (
        args.task_config
        if args.task_config is not None
        else TaskConfig(metadata_fields=tuple(args.metadata_fields))
    )
    grading = (
        GradingConfig.from_path(args.grading_config)
        if args.grading_config
        else None
    )
    assets = ClassificationAssets.from_files(
        registry=args.registry,
        corpus=args.corpus,
        task=task,
    )
    registry = assets.registry
    config = assets.task
    assert assets.corpus is not None
    corpus = assets.corpus
    rows = pq.read_table(args.data).to_pylist()
    stage1_rows = [r for r in rows if r["stage"] == "stage1"]

    # greedy, fixed decoding settings (identical for base and SFT), model loaded once
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForImageTextToText.from_pretrained(
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
        run_one(
            row,
            registry=registry,
            config=config,
            corpus=corpus,
            seed=args.seed,
            generate=hf_generate,
            grading=grading,
        )
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
                "ground_truth_level": o.ground_truth_level,
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
                "predicted_level": o.predicted_level,
                "leaf_correct": o.leaf_correct,
                "level_correct": o.level_correct,
                "stage2_completion": o.stage2_completion,
                "e2e_correct": o.e2e_correct,
                "failures": list(o.failures),
            }
            for o in outcomes
        ],
        "registry": str(args.registry),
        "corpus": str(args.corpus),
        "data": str(args.data),
        "grading_config": str(args.grading_config) if args.grading_config else None,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    path = Path(args.report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
