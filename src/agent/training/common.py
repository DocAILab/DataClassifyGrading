"""Shared helpers for training dataset adapters (SFT and RL).

Extracted from the SFT exporter so both adapters validate the canonical
contract identically without duplicating logic. These helpers are pure
task/contract logic: they contain no training-algorithm state and no VeRL
imports.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

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


def build_candidates(ground_truth: str, registry: LeafRegistry) -> list[str]:
    """Deterministic baseline/test fixture policy: GT followed by the first
    four non-GT registry IDs. This is a fixture, NOT the production Stage 1
    retrieval policy."""
    if ground_truth not in registry.ids:
        raise ValueError(f"ground-truth category_id is absent from leaf registry: {ground_truth}")
    return [ground_truth] + [category_id for category_id in registry.ids if category_id != ground_truth][:4]


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
