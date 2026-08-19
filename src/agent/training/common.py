"""Shared helpers for training dataset adapters (SFT and RL).

Extracted from the SFT exporter so both adapters validate the canonical
contract identically without duplicating logic. These helpers are pure
task/contract logic: they contain no training-algorithm state and no VeRL
imports.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

from agent.task.contracts import CorpusCategory, LeafRegistry


def canonical_target(
    item: Mapping[str, Any],
    index: int,
    source: Path,
    registry: LeafRegistry,
) -> str | None:
    """Return the canonical ground-truth category_id, or None for records that
    must not enter training (anything but resolution_status == "resolved").

    Raises on schema/consistency violations instead of silently falling back:
    resolved records must carry a target whose category_id belongs to the
    registry and whose leaf_name matches the registry category name.
    """
    status = str(item.get("resolution_status", "") or "").strip()
    if status != "resolved":
        return None
    target = item.get("target")
    if not isinstance(target, Mapping):
        raise ValueError(f"item {index} in {source} is resolved but has no target")
    category_id = str(target.get("category_id", "") or "").strip()
    if not category_id:
        raise ValueError(f"item {index} in {source} has an empty target.category_id")
    if category_id not in registry.ids:
        raise ValueError(
            f"item {index} in {source} target.category_id {category_id!r} "
            "is absent from the leaf registry"
        )
    leaf_name = str(target.get("leaf_name", "") or "").strip()
    expected_name = registry.get(category_id).name
    if leaf_name != expected_name:
        raise ValueError(
            f"item {index} in {source} target.leaf_name {leaf_name!r} does not "
            f"match registry category {category_id!r} name {expected_name!r}"
        )
    return category_id


def build_candidates(
    ground_truth: str,
    registry: LeafRegistry,
    *,
    source_id: str,
) -> list[str]:
    """Deterministic candidate bundle shared by SFT and RL for one sample.

    Baseline fixture policy (no hard negatives): the ground truth plus the
    first four non-GT registry ids, then permuted deterministically from the
    stable ``source_id``. The permutation keeps the bundle reproducible
    (same source_id always yields the same ordering, across runs and across
    the SFT/RL exporters) while preventing the ground truth from sitting at
    a fixed position — which would otherwise leak a systematic
    ``{"answer":"1"}`` bias into every Stage 2 sample. This is a fixture,
    NOT the production Stage 1 retrieval policy.
    """
    if ground_truth not in registry.ids:
        raise ValueError(
            f"ground-truth category_id is absent from leaf registry: {ground_truth}"
        )
    base = [ground_truth] + [
        category_id for category_id in registry.ids if category_id != ground_truth
    ][:4]
    return _permute(base, source_id)


def _permute(items: Sequence[str], source_id: str) -> list[str]:
    """Deterministic Fisher–Yates shuffle keyed by a stable sha256 digest.

    ``hash()`` is randomized per process (PYTHONHASHSEED) and must never be
    used as a reproducibility seed; sha256 over the UTF-8 source_id is
    stable across runs, machines and processes. There is no runtime
    randomness.
    """
    digest = hashlib.sha256(source_id.encode("utf-8")).digest()
    result = list(items)
    for index in range(len(result) - 1, 0, -1):
        swap = digest[index % len(digest)] % (index + 1)
        result[index], result[swap] = result[swap], result[index]
    return result


def require_corpus(corpus: Mapping[str, CorpusCategory]) -> Mapping[str, CorpusCategory]:
    if not corpus:
        raise ValueError(
            "corpus is required and must be non-empty: production Stage 2 has "
            "no registry fallback"
        )
    return corpus


def require_corpus_covers_registry(
    corpus: Mapping[str, CorpusCategory],
    registry: LeafRegistry,
) -> None:
    """Production invariant: the canonical corpus must define every registry
    category so Stage 2 can always resolve candidates by category_id (no
    fallback, no late surprise)."""
    missing = [category_id for category_id in registry.ids if category_id not in corpus]
    if missing:
        raise ValueError(
            "corpus is missing registry categories (Stage 2 cannot resolve "
            f"them by category_id): {sorted(missing)}"
        )


__all__ = [
    "canonical_target",
    "build_candidates",
    "require_corpus",
    "require_corpus_covers_registry",
]
