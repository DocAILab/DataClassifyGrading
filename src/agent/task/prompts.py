"""Deterministic two-stage HF-message prompts for leaf classification.

Prompt-facing identity is the choice protocol, never the canonical
category_id:
- Stage 1 receives the FULL LeafRegistry as candidate universe, rendered as
  compact [choice_id, display_name, standard_summary] entries. The summary is
  the registry description; full corpus descriptions/examples remain in
  Stage 2. The model answers with global choice ids ("1".."N" following
  registry order).
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

from .contracts import CorpusCategory, GradingConfig, LeafRegistry, TaskConfig
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
        "You are a leaf-category candidate retriever. Return exactly one JSON object "
        'with key "candidates". '
        "The value must contain exactly five unique choice ids from the catalog. "
        "Do not output Markdown, commentary, canonical category ids, or any other keys."
    )
    # Stage 1 must see the classification standard, not only opaque label
    # names. Registry descriptions are the compact standard summaries; full
    # corpus descriptions/examples remain in the five-candidate Stage 2
    # bundle. Canonical ids stay hidden behind choice ids.
    catalog = [
        [
            choice.choice_id,
            choice.display_name,
            registry.get(choice.category_id).description,
        ]
        for choice in choices.choices
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
    grading: GradingConfig | None = None,
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
    if grading is None:
        system = (
            "You are a leaf-category reranker. Return exactly one JSON object with key "
            '"answer". '
            'Its value must be one of the five candidate ids "1" through "5". '
            "Do not output Markdown, commentary, or any other keys."
        )
        rubric_block = ""
    else:
        system = (
            "You are a leaf-category reranker. Return exactly one JSON object with "
            'keys "answer" and "level". '
            '"answer" must be one of the five candidate ids "1" through "5". '
            '"level" must be exactly one of the listed sensitivity level codes. '
            "Do not output Markdown, commentary, or any other keys."
        )
        rubric_block = (
            "Sensitivity levels:\n"
            + json.dumps(
                [[code, text] for code, text in grading.rubric()],
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
        )
    user = (
        "Candidate bundle:\n"
        + json.dumps(bundle, ensure_ascii=False, separators=(",", ":"))
        + "\n"
        + rubric_block
        + "Field metadata:\n"
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


def stage2_answer(
    category_id: str,
    candidates: Sequence[str],
    *,
    level: str | None = None,
) -> str:
    """Assistant answer for Stage 2: canonical answer -> local bundle id.

    With ``level`` set (joint grading head) the JSON carries both keys;
    otherwise the strict single-key shape is emitted.
    """
    if category_id not in candidates:
        raise ValueError("stage2 answer must be one of the candidates")
    payload: dict[str, str] = {"answer": encode_stage2_answer(category_id, candidates)}
    if level is not None:
        payload["level"] = level
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
