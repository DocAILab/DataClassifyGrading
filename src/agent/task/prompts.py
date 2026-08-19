"""Deterministic two-stage HF-message prompts for leaf classification.

Prompt-facing identity is the choice protocol, never the canonical
category_id:
- Stage 1 receives the FULL LeafRegistry as candidate universe, rendered as
  compact [choice_id, display_name] pairs only (descriptions/examples are
  deliberately kept out of Stage 1). The model answers with global choice
  ids ("1".."N" following registry order).
- Stage 2 resolves candidates by canonical category_id against the
  canonical corpus (description/descriptions/examples) and renders each
  candidate with a LOCAL bundle id ("1".."5"); the model answers with the
  local id. When no corpus is provided it falls back to the registry
  description.

Decoding back to canonical category_id happens immediately at the LLM
boundary (evaluation adapters / SFT validator); choice ids never leak into
corpus lookup, canonical targets or reward semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Mapping, Sequence

from .contracts import CorpusCategory, LeafRegistry, TaskConfig
from .prompt_choices import PromptChoiceRegistry, encode_stage2_answer


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
    metadata: Mapping[str, object],
    registry: LeafRegistry,
    config: TaskConfig,
    choices: PromptChoiceRegistry | None = None,
) -> Prompt:
    choices = choices or PromptChoiceRegistry.from_registry(registry)
    system = (
        "You are a leaf-category candidate retriever. Return only one JSON object, "
        'with exactly this shape: {"candidates":["1","2","3","4","5"]}. '
        "The candidates array must contain exactly 5 unique choice ids from the "
        "catalog. "
        "Do not output Markdown, commentary, canonical category ids, or any other keys."
    )
    catalog = [
        [choice.choice_id, choice.display_name] for choice in choices.choices
    ]
    user = (
        "Retrieve five candidate leaf categories from this catalog:\n"
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
    choices: PromptChoiceRegistry | None = None,
) -> Prompt:
    if len(candidates) != 5 or len(set(candidates)) != 5:
        raise ValueError("stage2 requires exactly 5 unique candidates")
    if any(candidate not in registry.ids for candidate in candidates):
        raise ValueError("stage2 candidates must belong to the leaf registry")
    choices = choices or PromptChoiceRegistry.from_registry(registry)
    bundle = []
    for index, category_id in enumerate(candidates, start=1):
        display_name = choices.display_name_of(category_id)
        if corpus is not None:
            # resolve by category_id only; never by bare leaf name
            corpus_category = corpus.get(category_id)
            if corpus_category is None:
                raise ValueError(
                    f"stage2 candidate {category_id!r} is absent from the canonical corpus"
                )
            bundle.append(
                {
                    "id": str(index),
                    "name": display_name,
                    "description": corpus_category.description,
                    "descriptions": list(corpus_category.descriptions),
                    "examples": list(corpus_category.examples),
                }
            )
        else:
            category = registry.get(category_id)
            bundle.append(
                {
                    "id": str(index),
                    "name": display_name,
                    "description": category.description,
                }
            )
    system = (
        "You are a leaf-category reranker. Return only one JSON object, exactly "
        '{"answer":"1"}. The answer must be one of the five candidate ids 1-5. '
        "Do not output Markdown, commentary, or any other keys."
    )
    user = (
        "Candidate bundle:\n"
        + json.dumps(bundle, ensure_ascii=False, separators=(",", ":"))
        + "\nField metadata:\n"
        + _metadata_text(metadata, config)
    )
    return Prompt(system, user)


def stage1_answer(
    candidates: Sequence[str], *, choices: PromptChoiceRegistry
) -> str:
    """Assistant answer for Stage 1: canonical candidates -> global choice ids."""
    choice_ids = choices.encode_candidates(candidates)
    return json.dumps(
        {"candidates": list(choice_ids)},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def stage2_answer(category_id: str, candidates: Sequence[str]) -> str:
    """Assistant answer for Stage 2: canonical answer -> local bundle id."""
    if category_id not in candidates:
        raise ValueError("stage2 answer must be one of the candidates")
    return json.dumps(
        {"answer": encode_stage2_answer(category_id, candidates)},
        ensure_ascii=False,
        separators=(",", ":"),
    )
