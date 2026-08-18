"""Deterministic two-stage HF-message prompts for leaf classification.

Stage 1 receives the FULL LeafRegistry as candidate universe, rendered as
category_id + name pairs only (descriptions/examples are deliberately kept
out of Stage 1). Stage 2 resolves candidates by category_id against the
canonical corpus (category_id/name/description/descriptions/examples); when
no corpus is provided it falls back to the registry name/description.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Mapping, Sequence

from .contracts import CorpusCategory, LeafRegistry, TaskConfig


@dataclass(frozen=True)
class Prompt:
    system: str
    user: str


def _metadata_text(metadata: Mapping[str, object], config: TaskConfig) -> str:
    selected = {}
    for field in config.metadata_fields:
        value = metadata.get(field, "")
        selected[field] = "" if value is None else value
    return json.dumps(selected, ensure_ascii=False, separators=(",", ":"))


def build_stage1_prompt(
    metadata: Mapping[str, object], registry: LeafRegistry, config: TaskConfig
) -> Prompt:
    system = (
        "You are a leaf-category candidate retriever. Return only one JSON object, "
        'with exactly this shape: {"candidates":["category_id", "category_id", '
        '"category_id", "category_id", "category_id"]}. '
        "The candidates array must contain exactly 5 unique category_id values "
        "from the registry. "
        "Do not output Markdown, commentary, or any other keys."
    )
    catalog = [
        {"category_id": category.category_id, "name": category.name}
        for category in registry.categories
    ]
    user = (
        "Retrieve five candidate leaf categories from this registry:\n"
        + json.dumps(catalog, ensure_ascii=False)
        + "\nField metadata:\n"
        + _metadata_text(metadata, config)
    )
    return Prompt(system, user)


def build_stage2_prompt(
    metadata: Mapping[str, object],
    candidates: Sequence[str],
    registry: LeafRegistry,
    config: TaskConfig,
    corpus: Mapping[str, CorpusCategory] | None = None,
) -> Prompt:
    if len(candidates) != 5 or len(set(candidates)) != 5:
        raise ValueError("stage2 requires exactly 5 unique candidates")
    if any(candidate not in registry.ids for candidate in candidates):
        raise ValueError("stage2 candidates must belong to the leaf registry")
    bundle = []
    for category_id in candidates:
        if corpus is not None:
            # resolve by category_id only; never by bare leaf name
            corpus_category = corpus.get(category_id)
            if corpus_category is None:
                raise ValueError(
                    f"stage2 candidate {category_id!r} is absent from the canonical corpus"
                )
            bundle.append(
                {
                    "category_id": category_id,
                    "name": corpus_category.name,
                    "description": corpus_category.description,
                    "descriptions": list(corpus_category.descriptions),
                    "examples": list(corpus_category.examples),
                }
            )
        else:
            category = registry.get(category_id)
            bundle.append(
                {
                    "category_id": category_id,
                    "name": category.name,
                    "description": category.description,
                }
            )
    system = (
        "You are a leaf-category reranker. Return only one JSON object, exactly "
        '{"answer":"category_id"}. The answer must be one of the five candidates. '
        "Do not output Markdown, commentary, or any other keys."
    )
    user = (
        "Candidate bundle:\n"
        + json.dumps(bundle, ensure_ascii=False, separators=(",", ":"))
        + "\nField metadata:\n"
        + _metadata_text(metadata, config)
    )
    return Prompt(system, user)


def stage1_answer(candidates: Sequence[str]) -> str:
    if len(candidates) != 5 or len(set(candidates)) != 5:
        raise ValueError("stage1 requires exactly 5 unique candidates")
    return json.dumps({"candidates": list(candidates)}, ensure_ascii=False, separators=(",", ":"))


def stage2_answer(category_id: str, candidates: Sequence[str]) -> str:
    if category_id not in candidates:
        raise ValueError("stage2 answer must be one of the candidates")
    return json.dumps({"answer": category_id}, ensure_ascii=False, separators=(",", ":"))
