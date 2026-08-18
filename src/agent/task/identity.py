"""Category identity helpers for the canonical data contract.

Rules (from stage-1 report decisions and stage-2 review):
- category_id must be unique inside one dataset LeafRegistry and must not
  rely on random UUIDs or opaque hashes.
- Same leaf name under different parent paths must never collide: the
  deterministic path strategy qualifies the leaf with every path level, so
  structurally different paths always produce different category_ids.
- Path-based category_ids are human-readable and stable across runs and
  machines; whitespace is normalized (collapsed away) so label variants like
  "经营 管理" / "经营管理" resolve to the same identity, while genuinely
  different parents stay distinct.
- Codes (e.g. guanji A1-1-1) are treated as opaque stable identities; their
  digit groups are not interpreted as level_2 / level_3.
- A production LeafRegistry represents the complete leaf universe of a
  classification standard / corpus, never a set of training samples; the
  registry factory therefore consumes CorpusCategory, not SampleTarget.
"""

from __future__ import annotations

import re
from typing import Iterable, Sequence

from agent.task.contracts import CorpusCategory, LeafCategory, LeafRegistry

_WS_RE = re.compile(r"\s+")


def compact(text: str) -> str:
    """Remove every whitespace character (normalization for identity seeds)."""
    return _WS_RE.sub("", text)


def qualified_category_id(domain: str, path: Sequence[str]) -> str:
    """Deterministic, human-readable, path-qualified category_id.

    Format: ``<domain>:<part1>.<part2>.<part3>.<part4>`` where parts are the
    whitespace-collapsed path fields in fixed slot order (empty levels are
    kept as empty parts, so a missing level_3 remains distinguishable from a
    shorter path). The same leaf name under different parent paths therefore
    always yields different category_ids.
    """
    parts = [compact(part) for part in path]
    return f"{domain}:{'.'.join(parts)}"


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


def leaf_registry_from_corpus(
    categories: Iterable[CorpusCategory],
) -> LeafRegistry:
    """Build a LeafRegistry from the complete leaf universe of a corpus.

    The registry is derived from the classification standard / corpus, not
    from training samples, so it stays complete even for leaves that never
    appear in the training split. Descriptions may be missing per category
    (pers_info corpus is incomplete); LeafRegistry still enforces unique,
    non-empty category_ids and the minimum-category invariant.
    """
    leaf_categories = tuple(
        LeafCategory(
            category_id=category.category_id,
            name=category.name,
            description=category.description,
            path=category.path,
            code=category.code,
        )
        for category in categories
    )
    return LeafRegistry(leaf_categories)


__all__ = [
    "compact",
    "qualified_category_id",
    "code_leaf_map",
    "leaf_registry_from_corpus",
]
