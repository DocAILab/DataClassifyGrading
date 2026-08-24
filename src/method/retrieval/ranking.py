"""Validated stable ranking shared by all retrieval scorers."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def stable_rank(
    scores: np.ndarray | Sequence[float],
    registry_ids: Sequence[str],
    *,
    top_k: int,
) -> tuple[tuple[str, float], ...]:
    values = np.asarray(scores, dtype=np.float64)
    if values.shape != (len(registry_ids),) or not np.isfinite(values).all():
        raise ValueError("scores must be a finite vector aligned to registry IDs")
    if top_k < 1 or top_k > len(registry_ids):
        raise ValueError("top_k must be between one and registry size")
    order = np.lexsort((np.arange(len(registry_ids)), -values))[:top_k]
    return tuple((registry_ids[index], float(values[index])) for index in order)
