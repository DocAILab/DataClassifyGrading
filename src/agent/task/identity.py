"""Category identity helpers for the canonical data contract.

Rules (from stage-1 report decisions):
- category_id must be unique inside one dataset LeafRegistry and must not
  rely on random UUIDs.
- Same leaf name under different parent paths must never collide.
- Codes (e.g. guanji A1-1-1) are treated as opaque stable identities; their
  digit groups are not interpreted as level_2 / level_3.
- Deterministic path-hash IDs are stable across runs and machines; the hash
  seed keeps every path field (empty levels included) so that structurally
  different paths always produce different IDs.
"""

from __future__ import annotations

import hashlib
import re
from typing import Iterable, Mapping, Sequence

from agent.task.contracts import CorpusCategory, LeafCategory, LeafRegistry, SampleTarget

_WS_RE = re.compile(r"\s+")


def compact(text: str) -> str:
    """Remove every whitespace character (normalization for identity seeds)."""
    return _WS_RE.sub("", text)


def stable_category_id(domain: str, path: Sequence[str]) -> str:
    """Deterministic, collision-resistant category_id for one category path.

    The seed joins every path field (empty levels kept, so e.g. finance
    paths with a missing level_3 remain distinguishable from shorter paths)
    with the unit separator; the ID is ``<domain>:<hex16>``.
    """
    seed = "\x1f".join(compact(part) for part in path)
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]
    return f"{domain}:{digest}"


def code_leaf_map(
    categories: Iterable[CorpusCategory],
) -> dict[str, str]:
    """Map leaf name -> code for corpora that carry codes.

    Raises ValueError when the same leaf name maps to more than one distinct
    code (ambiguous identity; must be resolved before use).
    """
    mapping: dict[str, str] = {}
    for category in categories:
        if not category.code:
            continue
        leaf = category.name
        existing = mapping.get(leaf)
        if existing is not None and existing != category.code:
            raise ValueError(
                f"leaf name {leaf!r} maps to multiple codes: "
                f"{existing!r} vs {category.code!r}"
            )
        mapping[leaf] = category.code
    return mapping


def build_leaf_registry(
    targets: Iterable[SampleTarget],
    *,
    descriptions: Mapping[str, str] | None = None,
) -> LeafRegistry:
    """Build a LeafRegistry from resolved targets (deduplicated by category_id).

    description is optional per category: pers_info targets have no corpus
    description and still produce a valid registry entry.
    """
    by_id: dict[str, SampleTarget] = {}
    for target in targets:
        by_id.setdefault(target.category_id, target)
    categories = tuple(
        LeafCategory(
            category_id=target.category_id,
            description=(descriptions or {}).get(target.category_id, ""),
            name=target.leaf_name,
            path=target.category_path,
        )
        for target in by_id.values()
    )
    return LeafRegistry(categories)


__all__ = [
    "compact",
    "stable_category_id",
    "code_leaf_map",
    "build_leaf_registry",
]
