"""Train-only field priors and deterministic hybrid retrieval scoring."""

from __future__ import annotations

from collections import Counter, defaultdict
import math
import re
from typing import Any, Mapping, Sequence

import numpy as np


def normalize_field_name(value: object) -> str:
    """Normalize a field name without using its validation label."""
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _features(value: str) -> Counter[str]:
    normalized = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", " ", value.lower()).strip()
    compact = normalized.replace(" ", "")
    features: Counter[str] = Counter()
    for token in normalized.split():
        features[f"token:{token}"] += 5
    for size in (1, 2, 3):
        features.update(
            f"char{size}:{compact[index:index + size]}"
            for index in range(max(0, len(compact) - size + 1))
        )
    return features


def _cosine(left: Counter[str], right: Counter[str]) -> float:
    numerator = sum(value * right.get(key, 0) for key, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    return 0.0 if not left_norm or not right_norm else numerator / (left_norm * right_norm)


def build_field_label_index(
    records: Sequence[Mapping[str, Any]], registry_ids: Sequence[str]
) -> dict[str, Counter[str]]:
    """Build a field-name to level-4 count index from train records only."""
    allowed = set(registry_ids)
    index: dict[str, Counter[str]] = {}
    for record in records:
        metadata = record.get("metadata")
        classification = record.get("classification")
        if not isinstance(metadata, Mapping) or not isinstance(classification, Mapping):
            continue
        field_name = normalize_field_name(metadata.get("field_name"))
        label = str(classification.get("level_4", "")).strip()
        if field_name and label in allowed:
            index.setdefault(field_name, Counter())[label] += 1
    return index


def field_index_scores(
    field_names: Sequence[str],
    index: Mapping[str, Mapping[str, int]],
    registry_ids: Sequence[str],
    *,
    max_neighbors: int = 64,
) -> np.ndarray:
    """Score labels using exact train fields, then nearest train fields.

    Validation labels are never consulted.  For unseen fields, votes are
    transferred from the most similar train-seen field names using lexical
    feature cosine and log-count weighting.
    """
    if max_neighbors < 1:
        raise ValueError("max_neighbors must be positive")
    labels = tuple(registry_ids)
    if not labels or len(set(labels)) != len(labels):
        raise ValueError("registry_ids must be non-empty and unique")
    positions = {label: position for position, label in enumerate(labels)}
    normalized_index = {
        normalize_field_name(field): {
            label: int(count)
            for label, count in counts.items()
            if label in positions and int(count) > 0
        }
        for field, counts in index.items()
    }
    normalized_index = {field: counts for field, counts in normalized_index.items() if counts}
    field_features = {field: _features(field) for field in normalized_index}
    output = np.zeros((len(field_names), len(labels)), dtype=np.float64)
    for row, field_name in enumerate(field_names):
        query = normalize_field_name(field_name)
        exact = normalized_index.get(query)
        if exact:
            for label, count in exact.items():
                output[row, positions[label]] = math.log1p(count)
            continue
        if not field_features:
            continue
        neighbours = [
            (_cosine(_features(query), features), seen_field)
            for seen_field, features in field_features.items()
        ]
        neighbours = [item for item in neighbours if item[0] > 0]
        neighbours.sort(key=lambda item: (-item[0], item[1]))
        votes: Counter[str] = Counter()
        for similarity, seen_field in neighbours[:max_neighbors]:
            for label, count in normalized_index[seen_field].items():
                votes[label] += similarity * math.log1p(count)
        for label, score in votes.items():
            output[row, positions[label]] = float(score)
    return output


def _row_minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    minimum = values.min(axis=1, keepdims=True)
    maximum = values.max(axis=1, keepdims=True)
    span = maximum - minimum
    return np.divide(
        values - minimum,
        span,
        out=np.zeros_like(values, dtype=np.float64),
        where=span > 1e-12,
    )


def fuse_retrieval_scores(
    lexical_scores: np.ndarray,
    dense_scores: np.ndarray,
    index_scores: np.ndarray,
    *,
    lexical_weight: float = 0.25,
    dense_weight: float = 0.50,
    index_weight: float = 0.25,
) -> np.ndarray:
    """Fuse lexical, dense, and train-index scores after row normalization."""
    matrices = [
        np.asarray(lexical_scores, dtype=np.float64),
        np.asarray(dense_scores, dtype=np.float64),
        np.asarray(index_scores, dtype=np.float64),
    ]
    if any(matrix.ndim != 2 for matrix in matrices):
        raise ValueError("all score matrices must be rank-2")
    if any(matrix.shape != matrices[0].shape for matrix in matrices[1:]):
        raise ValueError("all score matrices must have the same shape")
    if any(not np.isfinite(matrix).all() for matrix in matrices):
        raise ValueError("score matrices must be finite")
    weights = (float(lexical_weight), float(dense_weight), float(index_weight))
    if any(weight < 0 or not math.isfinite(weight) for weight in weights):
        raise ValueError("fusion weights must be finite and non-negative")
    if sum(weights) <= 0:
        raise ValueError("at least one fusion weight must be positive")
    normalized = [_row_minmax(matrix) for matrix in matrices]
    return sum(weight * matrix for weight, matrix in zip(weights, normalized, strict=True))
