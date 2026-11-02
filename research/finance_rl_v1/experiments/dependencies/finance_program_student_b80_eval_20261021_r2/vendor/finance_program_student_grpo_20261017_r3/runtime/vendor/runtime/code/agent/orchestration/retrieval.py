"""Auditable Top-K retrieval backends for frozen standard registries."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from json import JSONDecodeError
import math
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence

from .runtime import KnowledgeBase


class RetrievalError(ValueError):
    """Raised when a retriever cannot return one valid frozen-registry ranking."""


@dataclass(frozen=True)
class RetrievalResult:
    candidates: tuple[dict[str, Any], ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)


_QUERY_FIELD_ORDER = (
    "domain",
    "field_name",
    "field_abbreviation_semantics",
    "field_semantics",
)


def format_retrieval_query(query: Any) -> str:
    """Render allowed retrieval evidence deterministically for text encoders."""

    if isinstance(query, Mapping):
        parts = [
            f"{name}: {query[name]}"
            for name in _QUERY_FIELD_ORDER
            if name in query and query[name] not in (None, "")
        ]
        return "\n".join(parts)
    return str(query)


def _canonical_registry(registry: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    canonical: dict[str, dict[str, Any]] = {}
    for key, raw in registry.items():
        category = str(raw.get("category") or raw.get("category_id") or key)
        if category in canonical:
            raise RetrievalError(f"duplicate Registry category: {category}")
        entry = dict(raw)
        entry["category"] = category
        canonical[category] = entry
    return canonical


def validate_ranked_candidates(
    candidates: Sequence[Mapping[str, Any]],
    registry: Mapping[str, Mapping[str, Any]],
    *,
    top_k: int,
) -> tuple[dict[str, Any], ...]:
    """Return authoritative Registry entries for one exact, distinct ranking."""

    if top_k < 1:
        raise RetrievalError("top_k must be positive")
    if len(candidates) != top_k:
        raise RetrievalError(f"ranking must contain exactly {top_k} candidates")
    frozen = _canonical_registry(registry)
    ranked: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, candidate in enumerate(candidates):
        category = candidate.get("category") or candidate.get("category_id")
        if not isinstance(category, str) or not category:
            raise RetrievalError(f"candidate {index} must contain a non-empty category")
        if category in seen:
            raise RetrievalError(f"duplicate category in ranking: {category}")
        if category not in frozen:
            raise RetrievalError(f"unknown Registry category: {category}")
        seen.add(category)
        ranked.append(dict(frozen[category]))
    return tuple(ranked)


class LexicalRetriever:
    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        registry: Mapping[str, Mapping[str, Any]],
    ) -> None:
        self.knowledge_base = knowledge_base
        self.registry = _canonical_registry(registry)

    def retrieve(self, query: Any, top_k: int = 5) -> RetrievalResult:
        candidates = self.knowledge_base.search_standard(query, top_k=top_k)
        return RetrievalResult(
            validate_ranked_candidates(candidates, self.registry, top_k=top_k),
            {"backend": "lexical"},
        )


def _single_json_object(text: str) -> Mapping[str, Any]:
    if not isinstance(text, str) or not text.strip():
        raise RetrievalError("model response must be non-empty JSON")
    cleaned = text.strip()
    if not cleaned.startswith("{"):
        raise RetrievalError("model response must be one JSON object")
    try:
        value, end = json.JSONDecoder().raw_decode(cleaned)
    except JSONDecodeError as exc:
        raise RetrievalError(f"model response is invalid JSON: {exc.msg}") from exc
    if cleaned[end:].strip() or not isinstance(value, Mapping):
        raise RetrievalError("model response must be exactly one JSON object")
    return value


def _last_generation_metadata(generate: Callable[..., Any]) -> dict[str, Any]:
    metadata = getattr(generate, "last_generation_metadata", None)
    return dict(metadata) if isinstance(metadata, Mapping) else {}


class CatalogModelRetriever:
    """Ask a text generator to rank IDs from the complete frozen Registry."""

    def __init__(
        self,
        generate: Callable[[list[dict[str, str]]], str],
        registry: Mapping[str, Mapping[str, Any]],
    ) -> None:
        self.generate = generate
        self.registry = _canonical_registry(registry)

    def _messages(self, query: Any, top_k: int) -> list[dict[str, str]]:
        catalog = [
            {
                "category_id": category,
                "description": str(entry.get("description") or entry.get("definition") or category),
            }
            for category, entry in self.registry.items()
        ]
        payload = {
            "semantic_query": query,
            "top_k": top_k,
            "category_catalog": catalog,
            "output_schema": {"category_ids": ["exact Registry category ID"]},
        }
        return [
            {
                "role": "system",
                "content": (
                    "Rank the database-field categories supported by the semantic query. "
                    "Choose only exact category_id values from the supplied complete catalog. "
                    f"Return exactly {top_k} distinct IDs in descending relevance. "
                    "Return one JSON object only, with exactly the key category_ids."
                ),
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]

    def retrieve(self, query: Any, top_k: int = 5) -> RetrievalResult:
        messages = self._messages(query, top_k)
        raw = self.generate(messages)
        payload = _single_json_object(raw)
        if set(payload) != {"category_ids"}:
            raise RetrievalError("model response must contain exactly category_ids")
        ids = payload.get("category_ids")
        if not isinstance(ids, list):
            raise RetrievalError("category_ids must be a list")
        candidates = [{"category": value} for value in ids]
        ranked = validate_ranked_candidates(candidates, self.registry, top_k=top_k)
        return RetrievalResult(
            ranked,
            {
                "backend": "model-catalog",
                "messages": messages,
                "raw_response": raw,
                **_last_generation_metadata(self.generate),
            },
        )


EmbeddingFunction = Callable[..., Sequence[Sequence[float]]]


def _normalized(vector: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(value) for value in vector)
    norm = math.sqrt(sum(value * value for value in values))
    if not values or norm == 0.0:
        raise RetrievalError("embedding vector must be non-empty and non-zero")
    return tuple(value / norm for value in values)


class EmbeddingRetriever:
    """Rank a frozen Registry with cached normalized catalog embeddings."""

    def __init__(
        self,
        encode: EmbeddingFunction,
        registry: Mapping[str, Mapping[str, Any]],
    ) -> None:
        self.encode = encode
        self.registry = _canonical_registry(registry)
        self._categories = tuple(self.registry)
        self._catalog_vectors: tuple[tuple[float, ...], ...] | None = None

    def _catalog_texts(self) -> list[str]:
        return [
            "passage: "
            + category
            + " "
            + str(entry.get("description") or entry.get("definition") or category)
            for category, entry in self.registry.items()
        ]

    def _ensure_catalog(self) -> tuple[tuple[float, ...], ...]:
        if self._catalog_vectors is None:
            encoded = self.encode(self._catalog_texts(), kind="passage")
            if len(encoded) != len(self._categories):
                raise RetrievalError("catalog embedding count does not match Registry")
            self._catalog_vectors = tuple(_normalized(vector) for vector in encoded)
            dimensions = {len(vector) for vector in self._catalog_vectors}
            if len(dimensions) != 1:
                raise RetrievalError("catalog embedding dimension mismatch")
        return self._catalog_vectors

    def retrieve(self, query: Any, top_k: int = 5) -> RetrievalResult:
        if top_k > len(self.registry):
            raise RetrievalError("top_k exceeds Registry size")
        started = perf_counter()
        catalog_vectors = self._ensure_catalog()
        query_rows = self.encode([f"query: {format_retrieval_query(query)}"], kind="query")
        if len(query_rows) != 1:
            raise RetrievalError("query encoder must return exactly one vector")
        query_vector = _normalized(query_rows[0])
        if len(query_vector) != len(catalog_vectors[0]):
            raise RetrievalError("query/catalog embedding dimension mismatch")
        scored = [
            (sum(left * right for left, right in zip(query_vector, vector)), category)
            for category, vector in zip(self._categories, catalog_vectors)
        ]
        scored.sort(key=lambda item: (-item[0], item[1]))
        ranked = validate_ranked_candidates(
            [{"category": category} for _, category in scored[:top_k]],
            self.registry,
            top_k=top_k,
        )
        return RetrievalResult(
            ranked,
            {
                "backend": "embedding",
                "elapsed_seconds": perf_counter() - started,
                "scores": [score for score, _ in scored[:top_k]],
            },
        )


class TransformersE5Encoder:
    """Lazy Hugging Face E5 encoder with masked mean pooling."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        local_files_only: bool = True,
        device_map: str = "auto",
        batch_size: int = 32,
    ) -> None:
        self.model_path = str(Path(model_path).expanduser())
        self.local_files_only = local_files_only
        self.device_map = device_map
        self.batch_size = batch_size
        self._tokenizer = None
        self._model = None
        self._torch = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional remote dependency
            raise RetrievalError("TransformersE5Encoder requires torch and transformers") from exc
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, local_files_only=self.local_files_only
        )
        self._model = AutoModel.from_pretrained(
            self.model_path,
            local_files_only=self.local_files_only,
            device_map=self.device_map,
            dtype="auto",
        )
        self._model.eval()

    def __call__(self, texts: Sequence[str], *, kind: str) -> list[list[float]]:
        if kind not in {"query", "passage"}:
            raise RetrievalError("embedding kind must be query or passage")
        self._load()
        assert self._tokenizer is not None
        assert self._model is not None
        assert self._torch is not None
        rows: list[list[float]] = []
        device = next(self._model.parameters()).device
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            inputs = self._tokenizer(
                batch,
                max_length=512,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            inputs = {name: value.to(device) for name, value in inputs.items()}
            with self._torch.no_grad():
                hidden = self._model(**inputs).last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            pooled = self._torch.nn.functional.normalize(pooled, p=2, dim=1)
            rows.extend(pooled.detach().cpu().tolist())
        return rows


__all__ = [
    "CatalogModelRetriever",
    "EmbeddingRetriever",
    "LexicalRetriever",
    "RetrievalError",
    "RetrievalResult",
    "TransformersE5Encoder",
    "format_retrieval_query",
    "validate_ranked_candidates",
]
