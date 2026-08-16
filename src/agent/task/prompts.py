"""Deterministic two-stage HF-message prompts for leaf classification."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Mapping, Sequence

from .contracts import LeafRegistry, TaskConfig


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
    user = (
        "Retrieve five candidate leaf categories from this registry:\n"
        + json.dumps(list(registry.ids), ensure_ascii=False)
        + "\nField metadata:\n"
        + _metadata_text(metadata, config)
    )
    return Prompt(system, user)


def build_stage2_prompt(
    metadata: Mapping[str, object],
    candidates: Sequence[str],
    registry: LeafRegistry,
    config: TaskConfig,
) -> Prompt:
    if len(candidates) != 5 or len(set(candidates)) != 5:
        raise ValueError("stage2 requires exactly 5 unique candidates")
    if any(candidate not in registry.ids for candidate in candidates):
        raise ValueError("stage2 candidates must belong to the leaf registry")
    bundle = [
        {"category_id": category_id, "description": registry.get(category_id).description}
        for category_id in candidates
    ]
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
