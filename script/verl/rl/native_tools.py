"""VeRL function-tool adapters for the deterministic category environment.

VeRL loads this module through ``multi_turn.function_tool_path``.  Runtime
asset paths are explicit launcher environment variables and the loaded index
is cached per reward/rollout worker.  The pure implementation remains CPU
importable and testable without VeRL.
"""

from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
from typing import Any

from agent.task import LeafRegistry
from agent.task.assets import load_corpus_categories
from agent.training.rl.native_tools import CategoryToolEnvironment

try:
    from verl.tools.function_tool import function_tool
except ModuleNotFoundError as exc:  # normal CPU/unit environment
    if exc.name != "verl":
        raise
    VERL_FUNCTION_TOOLS_AVAILABLE = False

    def function_tool(function):  # type: ignore[no-untyped-def]
        return function
else:
    VERL_FUNCTION_TOOLS_AVAILABLE = True

_REGISTRY_ENV = "DATACLASSIFY_RLOO_REGISTRY"
_CORPUS_ENV = "DATACLASSIFY_RLOO_CORPUS"


@lru_cache(maxsize=4)
def _load_environment(registry_path: str, corpus_path: str) -> CategoryToolEnvironment:
    registry = LeafRegistry.from_path(Path(registry_path))
    corpus = {
        item.category_id: item
        for item in load_corpus_categories(Path(corpus_path))
    }
    return CategoryToolEnvironment(registry, corpus)


def _runtime_environment() -> CategoryToolEnvironment:
    registry = os.environ.get(_REGISTRY_ENV, "").strip()
    corpus = os.environ.get(_CORPUS_ENV, "").strip()
    if not registry:
        raise RuntimeError(f"{_REGISTRY_ENV} is required for category tools")
    if not corpus:
        raise RuntimeError(f"{_CORPUS_ENV} is required for category tools")
    registry_path = Path(registry).resolve()
    corpus_path = Path(corpus).resolve()
    if not registry_path.is_file():
        raise FileNotFoundError(f"category tool registry not found: {registry_path}")
    if not corpus_path.is_file():
        raise FileNotFoundError(f"category tool corpus not found: {corpus_path}")
    return _load_environment(str(registry_path), str(corpus_path))


@function_tool
def search_categories(
    field_name: str,
    table_name: str,
    top_k: int = 5,
    scope: str = "",
) -> dict[str, Any]:
    """Search the approved category catalog using field and table metadata.

    Args:
        field_name: The source field or column name to classify.
        table_name: The source table name; use an empty string when unavailable.
        top_k: Number of candidates. The formal task fixes this value at five.
        scope: Optional level-1 group key from browse_categories; empty means all groups.
    """

    return _runtime_environment().search_categories(field_name, table_name, top_k, scope)


@function_tool
def get_category_details(choice_ids: list[str]) -> dict[str, Any]:
    """Get definitions for one or more opaque category choice ids.

    Args:
        choice_ids: One to five unique opaque choice ids returned by search.
    """

    return _runtime_environment().get_category_details(choice_ids)


@function_tool
def get_category_examples(
    choice_ids: list[str],
    limit: int = 2,
) -> dict[str, Any]:
    """Get bounded examples for one or more opaque category choice ids.

    Args:
        choice_ids: One to five unique opaque choice ids returned by search.
        limit: Maximum examples per category, from one through five.
    """

    return _runtime_environment().get_category_examples(choice_ids, limit)


@function_tool
def browse_categories(prefix: str = "") -> dict[str, Any]:
    """Browse the category catalog hierarchy.

    Args:
        prefix: Empty to list all level-1 groups with their scope keys;
            or a scope key to list its leaf categories.
    """

    return _runtime_environment().browse_categories(prefix or None)


__all__ = [
    "VERL_FUNCTION_TOOLS_AVAILABLE",
    "search_categories",
    "get_category_details",
    "get_category_examples",
    "browse_categories",
]
