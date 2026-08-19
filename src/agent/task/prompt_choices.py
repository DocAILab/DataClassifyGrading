"""Prompt-facing adapter between canonical category_ids and LLM action ids.

Boundary contract:
- canonical category_id stays the ONLY identity of the whole pipeline
  (registry / corpus / ground truth / evaluation / reward). It never changes.
- choice_id is a compact, deterministic numbering ("1", "2", ... following
  LeafRegistry.categories order) that exists ONLY inside prompts and model
  outputs. It is decoded back to category_id immediately at the LLM boundary
  and is never written back to canonical SampleTarget / CorpusCategory.
- display_name is the shortest unambiguous suffix of the category path: the
  leaf name when unique in the registry, otherwise parent-qualified until
  unique. No hashes, UUIDs or permanent encodings are introduced.

Stage 2 uses LOCAL bundle ids ("1".."5") in candidate order instead of the
global choice ids; decode is positional against the candidate bundle.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Sequence

from .contracts import LeafCategory, LeafRegistry


class PromptChoiceError(ValueError):
    """Raised when a choice id cannot be mapped to a canonical category_id."""


@dataclass(frozen=True)
class PromptChoice:
    """One prompt-facing entry: choice_id <-> category_id + display name."""

    choice_id: str
    category_id: str
    display_name: str


_STAGE2_LOCAL_IDS = tuple(str(index) for index in range(1, 6))


def _path_parts(category: LeafCategory) -> tuple[str, ...]:
    """Path parts with the leaf name guaranteed as the last element."""
    parts = list(category.path) if category.path else []
    if not parts or parts[-1] != category.name:
        parts = parts + [category.name]
    return tuple(parts)


def _display_names(registry: LeafRegistry) -> tuple[str, ...]:
    """Shortest unique path suffix per category; raises when impossible.

    A unique leaf name stays as-is. Duplicate leaf names are qualified with
    one parent level at a time until the suffix is unique against every
    other category; registries whose duplicates cannot be disambiguated
    (empty paths, identical full paths) fail explicitly instead of silently
    producing ambiguous display names.
    """
    counts = Counter(category.name for category in registry.categories)
    parts = {id(category): _path_parts(category) for category in registry.categories}
    names: list[str] = []
    for category in registry.categories:
        if counts[category.name] == 1:
            names.append(category.name)
            continue
        path = parts[id(category)]
        for depth in range(1, len(path) + 1):
            candidate = " / ".join(path[-depth:])
            if not any(
                other is not category
                and " / ".join(parts[id(other)][-depth:]) == candidate
                for other in registry.categories
            ):
                names.append(candidate)
                break
        else:
            raise PromptChoiceError(
                f"cannot build a unique display name for leaf {category.name!r} "
                f"(category_id {category.category_id!r}): duplicate leaf names "
                "cannot be disambiguated by path suffix"
            )
    if len(set(names)) != len(names):
        raise PromptChoiceError("display names must be unique across the registry")
    return tuple(names)


@dataclass(frozen=True)
class PromptChoiceRegistry:
    """Deterministic prompt-facing view over one LeafRegistry.

    choice ids are "1".."N" following LeafRegistry.categories order, so the
    mapping is stable across runs and identical registries. The registry is
    kept for coverage checks; canonical contracts are never modified.
    """

    registry: LeafRegistry
    choices: tuple[PromptChoice, ...]
    _by_choice_id: dict[str, PromptChoice] = field(
        init=False, compare=False, repr=False
    )
    _by_category_id: dict[str, PromptChoice] = field(
        init=False, compare=False, repr=False
    )

    def __post_init__(self) -> None:
        by_choice = {choice.choice_id: choice for choice in self.choices}
        by_category = {choice.category_id: choice for choice in self.choices}
        if len(by_choice) != len(self.choices):
            raise PromptChoiceError("prompt choice ids must be unique")
        if len(by_category) != len(self.choices):
            raise PromptChoiceError("prompt choices must not duplicate category ids")
        if set(by_category) != set(self.registry.ids):
            raise PromptChoiceError("prompt choices must cover the leaf registry")
        object.__setattr__(self, "_by_choice_id", by_choice)
        object.__setattr__(self, "_by_category_id", by_category)

    @classmethod
    def from_registry(cls, registry: LeafRegistry) -> "PromptChoiceRegistry":
        display_names = _display_names(registry)
        choices = tuple(
            PromptChoice(
                choice_id=str(index),
                category_id=category.category_id,
                display_name=display_name,
            )
            for index, (category, display_name) in enumerate(
                zip(registry.categories, display_names), start=1
            )
        )
        return cls(registry, choices)

    @property
    def choice_ids(self) -> tuple[str, ...]:
        return tuple(choice.choice_id for choice in self.choices)

    def contains_choice_id(self, choice_id: str) -> bool:
        return choice_id in self._by_choice_id

    def contains_category_id(self, category_id: str) -> bool:
        return category_id in self._by_category_id

    def choice_id_of(self, category_id: str) -> str:
        choice = self._by_category_id.get(category_id)
        if choice is None:
            raise PromptChoiceError(
                f"category_id {category_id!r} has no prompt choice"
            )
        return choice.choice_id

    def category_id_of(self, choice_id: str) -> str:
        choice = self._by_choice_id.get(choice_id)
        if choice is None:
            raise PromptChoiceError(
                f"choice id {choice_id!r} is not in the prompt catalog"
            )
        return choice.category_id

    def display_name_of(self, category_id: str) -> str:
        choice = self._by_category_id.get(category_id)
        if choice is None:
            raise PromptChoiceError(
                f"category_id {category_id!r} has no prompt choice"
            )
        return choice.display_name

    def encode_candidates(self, category_ids: Sequence[str]) -> tuple[str, ...]:
        """Canonical category_ids -> global choice ids (strict Stage 1 shape)."""
        if len(category_ids) != 5 or len(set(category_ids)) != 5:
            raise PromptChoiceError("stage1 requires exactly 5 unique candidates")
        return tuple(self.choice_id_of(category_id) for category_id in category_ids)

    def decode_candidates(self, choice_ids: Sequence[str]) -> tuple[str, ...]:
        """Global choice ids -> canonical category_ids (strict Stage 1 shape).

        Raises PromptChoiceError for anything but exactly 5 unique known
        choice ids; there is deliberately no name/fuzzy fallback.
        """
        if len(choice_ids) != 5:
            raise PromptChoiceError("stage1 prediction must contain exactly 5 candidates")
        if len(set(choice_ids)) != len(choice_ids):
            raise PromptChoiceError("stage1 candidates must be unique")
        return tuple(self.category_id_of(choice_id) for choice_id in choice_ids)


def encode_stage2_answer(category_id: str, candidates: Sequence[str]) -> str:
    """Canonical answer -> local bundle id ("1".."5") in candidate order."""
    if len(candidates) != 5:
        raise PromptChoiceError("stage2 requires exactly 5 candidates")
    try:
        return str(candidates.index(category_id) + 1)
    except ValueError:
        raise PromptChoiceError("stage2 answer must be one of the candidates") from None


def decode_stage2_answer(answer: str, candidates: Sequence[str]) -> str:
    """Local bundle id ("1".."5") -> canonical category_id.

    Raises PromptChoiceError for anything but an exact local id; there is
    deliberately no name/fuzzy fallback.
    """
    if len(candidates) != 5:
        raise PromptChoiceError("stage2 requires exactly 5 candidates")
    if answer not in _STAGE2_LOCAL_IDS:
        raise PromptChoiceError(f"stage2 answer {answer!r} must be one of 1..5")
    return candidates[int(answer) - 1]


__all__ = [
    "PromptChoice",
    "PromptChoiceError",
    "PromptChoiceRegistry",
    "encode_stage2_answer",
    "decode_stage2_answer",
]
