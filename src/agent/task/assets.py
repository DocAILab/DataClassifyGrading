"""Runtime-local classification asset loading.

The repository carries only synthetic examples. Production registries, corpora,
and task configuration are supplied as explicit local paths at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .contracts import CorpusCategory, LeafRegistry, TaskConfig


def load_corpus_categories(path: str | Path) -> tuple[CorpusCategory, ...]:
    """Load and validate a canonical corpus from an explicit local path."""
    source = Path(path)
    with source.open(encoding="utf-8") as handle:
        value = json.load(handle)
    raw = value.get("categories") if isinstance(value, Mapping) else value
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("corpus must contain a categories array")

    categories: list[CorpusCategory] = []
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise ValueError(f"invalid corpus category at index {index}")
        raw_code = item.get("code")
        categories.append(
            CorpusCategory(
                category_id=str(item.get("category_id", "")).strip(),
                name=str(item.get("name", "")).strip(),
                description=str(item.get("description", "") or "").strip(),
                descriptions=_strings(item.get("descriptions", ()), "descriptions", index),
                path=_strings(item.get("path", ()), "path", index),
                code=None if raw_code is None else str(raw_code).strip(),
                examples=_strings(item.get("examples", ()), "examples", index),
            )
        )
    ids = [category.category_id for category in categories]
    if len(ids) != len(set(ids)):
        raise ValueError("corpus category_id values must be unique")
    return tuple(categories)


def _strings(value: Any, field: str, index: int) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"corpus category {field} at index {index} must be an array")
    return tuple(str(item).strip() for item in value)


@dataclass(frozen=True)
class ClassificationAssets:
    """Validated runtime inputs for the two-stage classification interface."""

    registry: LeafRegistry
    task: TaskConfig
    corpus: Mapping[str, CorpusCategory] | None = None

    @classmethod
    def from_files(
        cls,
        *,
        registry: str | Path,
        task: str | Path | TaskConfig,
        corpus: str | Path | None = None,
    ) -> "ClassificationAssets":
        leaf_registry = LeafRegistry.from_path(registry)
        task_config = task if isinstance(task, TaskConfig) else TaskConfig.from_path(task)
        if corpus is None:
            return cls(leaf_registry, task_config, None)

        categories = load_corpus_categories(corpus)
        by_id = {category.category_id: category for category in categories}
        unknown = sorted(set(by_id) - set(leaf_registry.ids))
        if unknown:
            raise ValueError(
                "corpus contains category ids absent from the leaf registry: "
                + ", ".join(unknown)
            )
        return cls(leaf_registry, task_config, MappingProxyType(by_id))


__all__ = ["ClassificationAssets", "load_corpus_categories"]
