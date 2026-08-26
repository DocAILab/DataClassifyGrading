"""Deterministic category tools for native Qwen3.5 tool trajectories.

The tools deliberately use only the versioned leaf registry and canonical
corpus.  There is no network service, embedding model, vector store, training
label, or per-sample ground truth at this seam.  Canonical category ids remain
internal; model-facing calls use the registry's stable opaque choice ids.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
import unicodedata
from typing import Any, Mapping, Sequence

from agent.task.contracts import CorpusCategory, GradingConfig, LeafRegistry
from agent.task.prompt_choices import PromptChoiceRegistry

_TOOL_TOP_K = 5
_MAX_DETAIL_IDS = 5
_MAX_EXAMPLES = 5
_WS_RE = re.compile(r"\s+")
_ASCII_TOKEN_RE = re.compile(r"[a-z0-9]+")
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")


def _normalize(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("tool text arguments must be strings")
    return _WS_RE.sub(" ", unicodedata.normalize("NFKC", value).strip()).casefold()


def _features(value: str) -> frozenset[str]:
    """Return dependency-free lexical features for English and Chinese text."""

    normalized = _normalize(value)
    compact = normalized.replace(" ", "")
    features = set(_ASCII_TOKEN_RE.findall(normalized))
    features.update(_CJK_RE.findall(normalized))
    if len(compact) == 1:
        features.add(compact)
    else:
        features.update(compact[index : index + 2] for index in range(len(compact) - 1))
    return frozenset(item for item in features if item)


def _short(value: str, limit: int = 240) -> str:
    normalized = _WS_RE.sub(" ", value.strip())
    return normalized if len(normalized) <= limit else normalized[: limit - 1] + "…"


def _choice_ids(value: Sequence[str], choices: PromptChoiceRegistry) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("choice_ids must be an array of opaque choice ids")
    ids = tuple(value)
    if not ids or len(ids) > _MAX_DETAIL_IDS:
        raise ValueError(f"choice_ids must contain between 1 and {_MAX_DETAIL_IDS} ids")
    if any(not isinstance(item, str) or not item.strip() for item in ids):
        raise ValueError("choice_ids must contain non-empty strings")
    ids = tuple(item.strip() for item in ids)
    if len(set(ids)) != len(ids):
        raise ValueError("choice_ids must be unique")
    unknown = [item for item in ids if not choices.contains_choice_id(item)]
    if unknown:
        raise ValueError("unknown opaque choice id(s): " + ", ".join(unknown))
    return ids


@dataclass(frozen=True)
class _SearchDocument:
    choice_id: str
    category_id: str
    name: str
    summary: str
    name_text: str
    name_features: frozenset[str]
    body_features: frozenset[str]
    index: int


class CategoryToolEnvironment:
    """In-memory lexical search and read-only helpers over one category JSON set."""

    def __init__(
        self,
        registry: LeafRegistry,
        corpus: Mapping[str, CorpusCategory],
    ) -> None:
        if not isinstance(registry, LeafRegistry):
            raise ValueError("category tools require a LeafRegistry")
        if not isinstance(corpus, Mapping):
            raise ValueError("category tools require a canonical corpus mapping")
        missing = sorted(set(registry.ids) - set(corpus))
        extra = sorted(set(corpus) - set(registry.ids))
        if missing or extra:
            raise ValueError(
                "category tool corpus must exactly cover the registry; "
                f"missing={missing}, extra={extra}"
            )
        self.registry = registry
        self.corpus = dict(corpus)
        self.choices = PromptChoiceRegistry.from_registry(registry)
        self._canonical_ids = tuple(sorted(registry.ids, key=len, reverse=True))
        documents: list[_SearchDocument] = []
        for index, choice in enumerate(self.choices.choices):
            category = registry.get(choice.category_id)
            corpus_entry = self.corpus[choice.category_id]
            summary = corpus_entry.description or category.description
            name_parts = [choice.display_name, category.name, *category.path]
            body_parts = [
                summary,
                *corpus_entry.descriptions,
                *corpus_entry.examples,
                *corpus_entry.path,
            ]
            documents.append(
                _SearchDocument(
                    choice_id=choice.choice_id,
                    category_id=choice.category_id,
                    name=self._public_text(choice.display_name),
                    summary=_short(self._public_text(summary)),
                    name_text=_normalize(" ".join(part for part in name_parts if part)),
                    name_features=_features(" ".join(part for part in name_parts if part)),
                    body_features=_features(" ".join(part for part in body_parts if part)),
                    index=index,
                )
            )
        self._documents = tuple(documents)

    def _public_text(self, value: str) -> str:
        """Redact accidental canonical-id mentions from model-visible text."""

        result = value
        for category_id in self._canonical_ids:
            result = result.replace(category_id, "[redacted-id]")
        return result

    def search_categories(
        self,
        field_name: str,
        table_name: str,
        top_k: int = _TOOL_TOP_K,
    ) -> dict[str, Any]:
        """Return five deterministic lexical candidates without label access."""

        field = _normalize(field_name)
        table = _normalize(table_name)
        if not field:
            raise ValueError("field_name must be a non-empty string")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k != _TOOL_TOP_K:
            raise ValueError(f"top_k is fixed at {_TOOL_TOP_K}")
        field_features = _features(field)
        table_features = _features(table)
        query_features = field_features | table_features

        def score(document: _SearchDocument) -> tuple[float, int]:
            value = 0.0
            if field == document.name_text:
                value += 100.0
            elif field in document.name_text or document.name_text in field:
                value += 35.0
            value += 8.0 * len(field_features & document.name_features)
            value += 3.0 * len(table_features & document.name_features)
            value += 2.0 * len(query_features & document.body_features)
            return value, -document.index

        ranked = sorted(self._documents, key=score, reverse=True)[:top_k]
        return {
            "candidates": [
                {
                    "choice_id": document.choice_id,
                    "name": document.name,
                    "summary": document.summary,
                }
                for document in ranked
            ]
        }

    def get_category_details(self, choice_ids: Sequence[str]) -> dict[str, Any]:
        """Return descriptions for opaque ids, preserving caller order."""

        ids = _choice_ids(choice_ids, self.choices)
        details = []
        for choice_id in ids:
            category_id = self.choices.category_id_of(choice_id)
            corpus_entry = self.corpus[category_id]
            details.append(
                {
                    "choice_id": choice_id,
                    "name": self._public_text(
                        self.choices.display_name_of(category_id)
                    ),
                    "description": self._public_text(corpus_entry.description),
                    "descriptions": [
                        self._public_text(value)
                        for value in corpus_entry.descriptions
                    ],
                }
            )
        return {"categories": details}

    def get_category_examples(
        self,
        choice_ids: Sequence[str],
        limit: int = 2,
    ) -> dict[str, Any]:
        """Return bounded examples for opaque ids, preserving caller order."""

        ids = _choice_ids(choice_ids, self.choices)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= _MAX_EXAMPLES:
            raise ValueError(f"limit must be an integer between 1 and {_MAX_EXAMPLES}")
        examples = []
        for choice_id in ids:
            category_id = self.choices.category_id_of(choice_id)
            corpus_entry = self.corpus[category_id]
            examples.append(
                {
                    "choice_id": choice_id,
                    "name": self._public_text(
                        self.choices.display_name_of(category_id)
                    ),
                    "examples": [
                        self._public_text(value)
                        for value in corpus_entry.examples[:limit]
                    ],
                }
            )
        return {"categories": examples}


@dataclass(frozen=True)
class FinalToolAnswer:
    choice_id: str
    category_id: str
    level: str


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"final answer contains duplicate key {key!r}")
        value[key] = item
    return value


def parse_final_tool_answer(
    text: str,
    *,
    registry: LeafRegistry,
    grading: GradingConfig,
) -> FinalToolAnswer:
    """Decode one strict terminal assistant JSON object.

    The complete assistant segment must be JSON; Markdown, commentary,
    tool-call wrappers, unknown keys, canonical ids and invalid levels fail.
    """

    if not isinstance(text, str) or not text.strip():
        raise ValueError("final answer must be a non-empty JSON object")
    try:
        value = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("final answer must be exactly one JSON object") from exc
    if not isinstance(value, Mapping) or set(value) != {"answer", "level"}:
        raise ValueError("final answer must contain exactly answer and level")
    answer = value.get("answer")
    level = value.get("level")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("final answer.answer must be a non-empty opaque choice id")
    if not isinstance(level, str) or not level.strip():
        raise ValueError("final answer.level must be a non-empty level code")
    choice_id = answer.strip()
    level_code = level.strip()
    choices = PromptChoiceRegistry.from_registry(registry)
    if not choices.contains_choice_id(choice_id):
        raise ValueError("final answer.answer is not a known opaque choice id")
    if level_code not in grading.levels:
        raise ValueError("final answer.level is outside the approved rubric")
    return FinalToolAnswer(
        choice_id=choice_id,
        category_id=choices.category_id_of(choice_id),
        level=level_code,
    )


def exact_tool_reward(
    text: str,
    *,
    ground_truth: str,
    ground_truth_level: str,
    registry: LeafRegistry,
    grading: GradingConfig,
) -> float:
    """Return strict joint exact-match reward for the terminal assistant turn."""

    try:
        answer = parse_final_tool_answer(text, registry=registry, grading=grading)
    except (TypeError, ValueError):
        return 0.0
    return float(
        answer.category_id == ground_truth and answer.level == ground_truth_level
    )


__all__ = [
    "CategoryToolEnvironment",
    "FinalToolAnswer",
    "parse_final_tool_answer",
    "exact_tool_reward",
]
