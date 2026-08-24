"""Paired constrained Stage 2 evaluation for SFT and DPO policies."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

from agent.task import LeafRegistry, TaskConfig
from agent.task.prompts import build_stage2_prompt
from agent.task.prompts import stage2_answer
from method.dpo.label_scoring import retrieve_semantic_hard_negatives


def build_evaluation_case(
    record: Mapping[str, Any], registry: LeafRegistry, *, seed: int = 137
) -> dict[str, Any]:
    """Build one deterministic oracle Stage 2 case from val-visible facts."""
    source_id = str(record.get("id", "")).strip()
    metadata = record.get("metadata")
    classification = record.get("classification")
    if not source_id or not isinstance(metadata, Mapping) or not isinstance(classification, Mapping):
        raise ValueError("evaluation record has invalid id, metadata or classification")
    field_name = str(metadata.get("field_name", "") or "")
    golden = str(classification.get("level_4", "")).strip()
    negatives = retrieve_semantic_hard_negatives(field_name, golden, registry, count=4)
    candidates = [golden, *negatives]
    digest = hashlib.sha256(f"{seed}\0{source_id}".encode("utf-8")).digest()
    random.Random(int.from_bytes(digest[:16], "big")).shuffle(candidates)
    config = TaskConfig(("field_name",), "field_name_level4")
    prompt = build_stage2_prompt(
        {"field_name": field_name}, candidates, registry, config
    )
    return {
        "source_id": source_id,
        "ground_truth": golden,
        "candidates": candidates,
        "prompt": [
            {"role": "system", "content": prompt.system},
            {"role": "user", "content": prompt.user},
        ],
        "metadata": {"field_name": field_name},
        "candidate_policy": "semantic_hard_oracle_shuffled_v1",
        "seed": seed,
        "real_test_split_read": False,
    }


def prediction_from_scores(
    case: Mapping[str, Any], scores: Mapping[str, float]
) -> dict[str, Any]:
    candidates = list(case.get("candidates", []))
    if len(candidates) != 5 or len(set(candidates)) != 5:
        raise ValueError("evaluation case requires five unique candidates")
    if set(scores) != set(candidates):
        raise ValueError("candidate score keys do not match evaluation candidates")
    numeric = {label: float(scores[label]) for label in candidates}
    if any(not math.isfinite(value) for value in numeric.values()):
        raise ValueError("candidate scores must be finite")
    prediction = max(candidates, key=lambda label: numeric[label])
    golden = str(case.get("ground_truth", ""))
    return {
        "source_id": case.get("source_id"),
        "ground_truth": golden,
        "prediction": prediction,
        "candidates": candidates,
        "scores": numeric,
        "raw_output": json.dumps(
            {"answer": prediction}, ensure_ascii=False, separators=(",", ":")
        ),
        "format_valid": True,
        "contract_valid": prediction in candidates,
        "correct": prediction == golden,
        "candidate_policy": case.get("candidate_policy"),
        "seed": case.get("seed"),
        "real_test_split_read": False,
    }


def _load_prediction_rows(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            source_id = str(row.get("source_id", "")).strip()
            if not source_id or source_id in rows:
                raise ValueError(f"prediction source_id must be unique at line {line_number}")
            rows[source_id] = row
    return rows


def evaluate_cases(
    cases: Sequence[Mapping[str, Any]],
    output_path: str | Path,
    *,
    score_fn,
) -> dict[str, int]:
    """Score cases with per-source durable resume."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = _load_prediction_rows(output)
    existing = len(completed)
    new = 0
    seen: set[str] = set()
    for case in cases:
        source_id = str(case.get("source_id", "")).strip()
        if not source_id or source_id in seen:
            raise ValueError("evaluation cases require unique source_id values")
        seen.add(source_id)
        if source_id in completed:
            continue
        candidates = list(case.get("candidates", []))
        answers = {label: stage2_answer(label, candidates) for label in candidates}
        scores = score_fn(case.get("prompt", []), answers)
        row = prediction_from_scores(case, scores)
        with output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        completed[source_id] = row
        new += 1
    extras = set(completed) - seen
    if extras:
        raise ValueError(f"prediction rows do not match requested cases: {sorted(extras)[:5]}")
    return {"existing_rows": existing, "new_rows": new, "total_rows": len(completed)}


def _index_unique(rows: Sequence[Mapping[str, Any]], name: str) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        source_id = str(row.get("source_id", "")).strip()
        if not source_id or source_id in indexed:
            raise ValueError(f"{name} source_id values must be non-empty and unique")
        indexed[source_id] = row
    return indexed


def _policy_metrics(rows: Sequence[Mapping[str, Any]], labels: Sequence[str]) -> dict[str, Any]:
    label_set = set(labels)
    total = len(rows)
    correct = sum(
        row.get("contract_valid") is True
        and row.get("prediction") == row.get("ground_truth")
        for row in rows
    )
    invalid = sum(row.get("format_valid") is not True for row in rows)
    oov = sum(row.get("prediction") not in label_set for row in rows)
    f1_values: list[float] = []
    per_label: dict[str, dict[str, float | int]] = {}
    for label in labels:
        tp = sum(row.get("ground_truth") == label and row.get("prediction") == label for row in rows)
        fp = sum(row.get("ground_truth") != label and row.get("prediction") == label for row in rows)
        fn = sum(row.get("ground_truth") == label and row.get("prediction") != label for row in rows)
        support = sum(row.get("ground_truth") == label for row in rows)
        denominator = 2 * tp + fp + fn
        f1 = 2 * tp / denominator if denominator else 0.0
        recall = tp / support if support else 0.0
        f1_values.append(f1)
        per_label[label] = {
            "support": support,
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "recall": recall,
            "f1": f1,
        }
    return {
        "accuracy": correct / total if total else 0.0,
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else 0.0,
        "correct": correct,
        "invalid": invalid,
        "oov": oov,
        "contract_invalid": sum(row.get("contract_valid") is not True for row in rows),
        "per_label": per_label,
    }


def _mcnemar_exact(left_only: int, right_only: int) -> float:
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    lower = min(left_only, right_only)
    tail = sum(math.comb(discordant, value) for value in range(lower + 1)) / (2 ** discordant)
    return min(1.0, 2.0 * tail)


def paired_classification_report(
    sft_rows: Sequence[Mapping[str, Any]],
    dpo_rows: Sequence[Mapping[str, Any]],
    labels: Sequence[str],
) -> dict[str, Any]:
    """Compare policies only after proving identical IDs, labels and candidates."""
    if not labels or len(labels) != len(set(labels)):
        raise ValueError("labels must be unique and non-empty")
    sft = _index_unique(sft_rows, "SFT")
    dpo = _index_unique(dpo_rows, "DPO")
    if set(sft) != set(dpo):
        raise ValueError("SFT and DPO source_id sets differ")
    ordered_ids = [str(row["source_id"]).strip() for row in sft_rows]
    paired: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for source_id in ordered_ids:
        left, right = sft[source_id], dpo[source_id]
        if left.get("ground_truth") != right.get("ground_truth"):
            raise ValueError(f"paired ground_truth differs: {source_id}")
        if left.get("candidates") != right.get("candidates"):
            raise ValueError(f"paired candidates differ: {source_id}")
        paired.append((left, right))
    sft_ordered = [left for left, _ in paired]
    dpo_ordered = [right for _, right in paired]
    sft_metrics = _policy_metrics(sft_ordered, labels)
    dpo_metrics = _policy_metrics(dpo_ordered, labels)
    sft_wrong_dpo_correct = 0
    sft_correct_dpo_wrong = 0
    both_correct = 0
    both_wrong = 0
    for left, right in paired:
        golden = left.get("ground_truth")
        left_correct = left.get("contract_valid") is True and left.get("prediction") == golden
        right_correct = right.get("contract_valid") is True and right.get("prediction") == golden
        if left_correct and right_correct:
            both_correct += 1
        elif left_correct:
            sft_correct_dpo_wrong += 1
        elif right_correct:
            sft_wrong_dpo_correct += 1
        else:
            both_wrong += 1
    per_class = {
        label: {
            "support": sft_metrics["per_label"][label]["support"],
            "sft_recall": sft_metrics["per_label"][label]["recall"],
            "sft_f1": sft_metrics["per_label"][label]["f1"],
            "dpo_recall": dpo_metrics["per_label"][label]["recall"],
            "dpo_f1": dpo_metrics["per_label"][label]["f1"],
        }
        for label in labels
    }
    sft_metrics.pop("per_label")
    dpo_metrics.pop("per_label")
    return {
        "evaluation": "constrained_stage2_same_hard_candidates",
        "rows": len(paired),
        "sft": sft_metrics,
        "dpo": dpo_metrics,
        "delta": {
            "accuracy": dpo_metrics["accuracy"] - sft_metrics["accuracy"],
            "macro_f1": dpo_metrics["macro_f1"] - sft_metrics["macro_f1"],
        },
        "paired": {
            "sft_wrong_dpo_correct": sft_wrong_dpo_correct,
            "sft_correct_dpo_wrong": sft_correct_dpo_wrong,
            "both_correct": both_correct,
            "both_wrong": both_wrong,
            "mcnemar_exact_p_value": _mcnemar_exact(
                sft_wrong_dpo_correct, sft_correct_dpo_wrong
            ),
        },
        "per_class": per_class,
    }
