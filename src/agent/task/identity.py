"""Deterministic category-identity helpers for the canonical contract.

Path identities qualify a leaf with every configured level and normalize
whitespace. Locally supplied codes remain opaque. A production LeafRegistry
represents a complete classification universe, never a training sample set.
"""

from __future__ import annotations

import re
import uuid
from typing import Any, Iterable, Mapping, Sequence

from agent.task.contracts import CorpusCategory, LeafCategory, LeafRegistry

_WS_RE = re.compile(r"\s+")

# The metadata fields are deliberately kept independent of the prompt/task
# layer. They identify one source record even when the input file or row
# ordering changes.
RECORD_ID_FIELDS = ("database_name", "table_name", "field_name")
_RECORD_ID_SEPARATOR = "\x1f"


def _identity_text(value: Any) -> str:
    """Normalize one metadata component for the stable identity seed."""
    if value is None:
        return ""
    return _WS_RE.sub(" ", str(value).strip())


def stable_record_id(dataset: str, metadata: Mapping[str, Any]) -> str:
    """Return the stable UUID5 identity for one processed metadata record.

    The seed is exactly ``dataset|database_name|table_name|field_name`` using
    the non-printing unit-separator between components. Source filenames and
    row positions are intentionally absent, so moving a file or reordering
    rows cannot change an existing record's identity.
    """
    normalized_dataset = _identity_text(dataset)
    if not normalized_dataset:
        raise ValueError("dataset must be a non-empty name")
    if not isinstance(metadata, Mapping):
        raise ValueError("metadata must be an object for stable record identity")
    missing = [field for field in RECORD_ID_FIELDS if field not in metadata]
    if missing:
        raise ValueError(
            "metadata missing identity field(s): " + ", ".join(missing)
        )
    seed = _RECORD_ID_SEPARATOR.join(
        [normalized_dataset, *(_identity_text(metadata[field]) for field in RECORD_ID_FIELDS)]
    )
    return str(uuid.uuid5(uuid.NAMESPACE_URL, seed))


def validate_record_id(
    dataset: str,
    record: Mapping[str, Any],
    *,
    index: int | None = None,
) -> str:
    """Validate a processed record's supplied ``id`` against its metadata.

    Canonical construction must not silently carry an id copied from a stale
    dataset/version. A missing or mismatched id therefore fails fast and the
    expected id is returned for callers that need it in diagnostics.
    """
    if not isinstance(record, Mapping):
        location = f" at index {index}" if index is not None else ""
        raise ValueError(f"record{location} must be an object")
    metadata = record.get("metadata")
    expected = stable_record_id(dataset, metadata)
    supplied = record.get("id")
    location = f" at index {index}" if index is not None else ""
    if not isinstance(supplied, str) or not supplied.strip():
        raise ValueError(f"missing record id{location}; expected {expected}")
    if supplied != expected:
        raise ValueError(
            f"stale record id{location}: supplied {supplied!r}, expected {expected!r}"
        )
    return expected


# More descriptive spelling for callers at the canonical boundary.
validate_record_identity = validate_record_id


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
    while LeafRegistry still enforces unique, non-empty category ids and the
    minimum-category invariant.
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
    "RECORD_ID_FIELDS",
    "stable_record_id",
    "validate_record_id",
    "validate_record_identity",
    "compact",
    "qualified_category_id",
    "code_leaf_map",
    "leaf_registry_from_corpus",
]
