"""Exact Stage 1 retrieval metrics."""

from __future__ import annotations

from collections import defaultdict
from statistics import mean, median
from typing import Any, Mapping, Sequence


def summarize_retrieval(
    rows: Sequence[Mapping[str, Any]],
    *,
    registry_ids: Sequence[str],
    train_counts: Mapping[str, int],
) -> dict[str, Any]:
    if not rows:
        raise ValueError("prediction rows must not be empty")
    registry = tuple(registry_ids)
    allowed = set(registry)
    seen: set[str] = set()
    per_class: dict[str, list[int]] = defaultdict(list)
    ranks: list[int] = []
    top5_labels: set[str] = set()
    top5_tuples: set[tuple[str, ...]] = set()
    duplicate_candidates = oov_candidates = invalid_rows = 0
    for row in rows:
        source_id = str(row.get("source_id", "")).strip()
        if not source_id or source_id in seen:
            raise ValueError(f"duplicate source_id or empty source_id: {source_id}")
        seen.add(source_id)
        gold = str(row.get("golden_level_4", "")).strip()
        if gold not in allowed:
            raise ValueError(f"gold label is absent from registry: {gold}")
        ranked = [str(value) for value in row.get("ranked_labels", [])]
        if not ranked:
            invalid_rows += 1
            continue
        duplicate_candidates += len(ranked) - len(set(ranked))
        oov_candidates += sum(value not in allowed for value in ranked)
        if gold not in ranked:
            invalid_rows += 1
            continue
        rank = ranked.index(gold) + 1
        ranks.append(rank)
        per_class[gold].append(rank)
        top5 = tuple(ranked[:5])
        top5_labels.update(top5)
        top5_tuples.add(top5)
    if len(ranks) != len(rows):
        raise ValueError("every row must contain a valid full-registry gold rank")

    def recall_at(k: int) -> float:
        return sum(rank <= k for rank in ranks) / len(ranks)

    def macro_at(k: int) -> float:
        return mean(sum(rank <= k for rank in values) / len(values) for values in per_class.values())

    per_class_report = {
        label: {
            "support": len(values),
            "train_count": int(train_counts.get(label, 0)),
            "recall_at_1": sum(rank <= 1 for rank in values) / len(values),
            "recall_at_3": sum(rank <= 3 for rank in values) / len(values),
            "recall_at_5": sum(rank <= 5 for rank in values) / len(values),
        }
        for label, values in per_class.items()
    }
    return {
        "rows": len(rows),
        "recall_at_1": recall_at(1),
        "recall_at_3": recall_at(3),
        "recall_at_5": recall_at(5),
        "macro_recall_at_1": macro_at(1),
        "macro_recall_at_3": macro_at(3),
        "macro_recall_at_5": macro_at(5),
        "macro_class_count": len(per_class),
        "mrr": mean(1.0 / rank for rank in ranks),
        "mean_gold_rank": mean(ranks),
        "median_gold_rank": median(ranks),
        "registry_top5_coverage_count": len(top5_labels),
        "registry_top5_coverage_fraction": len(top5_labels) / len(registry),
        "unique_top5_tuples": len(top5_tuples),
        "duplicate_candidates": duplicate_candidates,
        "oov_candidates": oov_candidates,
        "invalid_rows": invalid_rows,
        "per_class": per_class_report,
    }
