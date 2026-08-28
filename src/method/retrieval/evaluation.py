"""Val-only Stage 1 retrieval evaluation and artifact production."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np

from agent.task import LeafRegistry
from method.sft.dataset import load_json_records

from .bge_m3 import dense_scores
from .char_ngram import char_ngram_scores
from .corpus import RegistryDocument, build_registry_documents
from .hybrid import build_field_label_index, field_index_scores, fuse_retrieval_scores
from .metrics import summarize_retrieval
from .ranking import stable_rank


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def _corpus_checksum(documents: Sequence[RegistryDocument]) -> str:
    payload = "\n".join(f"{item.category_id}\0{item.checksum}" for item in documents)
    return sha256(payload.encode("utf-8")).hexdigest()


def _validated_cases(records: Sequence[Mapping[str, Any]], registry: LeafRegistry) -> list[dict[str, str]]:
    cases: list[dict[str, str]] = []
    seen: set[str] = set()
    for record in records:
        source_id = str(record.get("id", "")).strip()
        metadata = record.get("metadata")
        classification = record.get("classification")
        if not source_id or source_id in seen:
            raise ValueError("validation source_id values must be unique and non-empty")
        if not isinstance(metadata, Mapping) or not isinstance(classification, Mapping):
            raise ValueError(f"invalid validation record: {source_id}")
        field_name = str(metadata.get("field_name", "") or "").strip()
        golden = str(classification.get("level_4", "")).strip()
        if not field_name:
            raise ValueError(f"validation record has empty field_name: {source_id}")
        if golden not in registry.ids:
            raise ValueError(f"gold label is absent from registry: {source_id}")
        seen.add(source_id)
        cases.append({"source_id": source_id, "field_name": field_name, "golden_level_4": golden})
    return cases


def _prediction_rows(
    cases: Sequence[Mapping[str, str]],
    score_matrix: np.ndarray,
    registry: LeafRegistry,
    *,
    method: str,
    corpus_checksum: str,
) -> list[dict[str, Any]]:
    if score_matrix.shape != (len(cases), len(registry.ids)):
        raise ValueError("score matrix shape does not match cases and registry")
    rows: list[dict[str, Any]] = []
    for case, scores in zip(cases, score_matrix, strict=True):
        ranked = stable_rank(scores, registry.ids, top_k=len(registry.ids))
        labels = [label for label, _ in ranked]
        top5 = ranked[:5]
        gold_rank = labels.index(case["golden_level_4"]) + 1
        rows.append({
            "source_id": case["source_id"],
            "field_name": case["field_name"],
            "golden_level_4": case["golden_level_4"],
            "top5": [label for label, _ in top5],
            "scores": [score for _, score in top5],
            "ranked_labels": labels,
            "gold_rank": gold_rank,
            "hit_at_1": gold_rank <= 1,
            "hit_at_3": gold_rank <= 3,
            "hit_at_5": gold_rank <= 5,
            "duplicate_candidates": len(labels) != len(set(labels)),
            "oov_candidates": any(label not in registry.ids for label in labels),
            "valid": True,
            "retrieval_method": method,
            "corpus_checksum": corpus_checksum,
            "metadata_fields": ["field_name"],
        })
    return rows


def evaluate_stage1(
    input_dir: str | Path,
    registry: LeafRegistry,
    output_dir: str | Path,
    *,
    encoder,
    limit: int | None = None,
    lexical_weight: float = 0.20,
    dense_weight: float = 0.50,
    index_weight: float = 0.30,
) -> dict[str, Any]:
    started = time.monotonic()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        torch = None
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    input_root = Path(input_dir)
    train_records = load_json_records(input_root / "train.json")
    records = load_json_records(input_root / "val.json")
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        records = records[:limit]
    cases = _validated_cases(records, registry)
    candidate_index = build_field_label_index(train_records, registry.ids)
    documents = build_registry_documents(registry)
    corpus_texts = [item.text for item in documents]
    queries = [item["field_name"] for item in cases]
    corpus_checksum = _corpus_checksum(documents)

    encode_started = time.monotonic()
    corpus_vectors = encoder.encode_corpus(corpus_texts)
    query_vectors = encoder.encode_queries(queries)
    bge_scores = dense_scores(query_vectors, corpus_vectors)
    encode_elapsed = time.monotonic() - encode_started
    np.savez_compressed(
        output / "registry_embeddings.npz",
        category_ids=np.asarray(registry.ids),
        vectors=np.asarray(corpus_vectors, dtype=np.float32),
        corpus_checksum=np.asarray(corpus_checksum),
    )
    lexical_started = time.monotonic()
    lexical_scores = char_ngram_scores(queries, corpus_texts)
    lexical_elapsed = time.monotonic() - lexical_started
    index_started = time.monotonic()
    index_scores = field_index_scores(queries, candidate_index, registry.ids)
    index_elapsed = time.monotonic() - index_started
    hybrid_scores = fuse_retrieval_scores(
        lexical_scores,
        bge_scores,
        index_scores,
        lexical_weight=lexical_weight,
        dense_weight=dense_weight,
        index_weight=index_weight,
    )

    bge_rows = _prediction_rows(cases, bge_scores, registry, method="bge-m3-dense-cls-normalized", corpus_checksum=corpus_checksum)
    char_rows = _prediction_rows(cases, lexical_scores, registry, method="char-ngram-v1", corpus_checksum=corpus_checksum)
    index_rows = _prediction_rows(cases, index_scores, registry, method="train-field-index-v1", corpus_checksum=corpus_checksum)
    hybrid_rows = _prediction_rows(cases, hybrid_scores, registry, method="hybrid-char-bge-train-index-v1", corpus_checksum=corpus_checksum)
    bge_metrics = summarize_retrieval(bge_rows, registry_ids=registry.ids, train_counts={})
    char_metrics = summarize_retrieval(char_rows, registry_ids=registry.ids, train_counts={})
    index_metrics = summarize_retrieval(index_rows, registry_ids=registry.ids, train_counts={})
    hybrid_metrics = summarize_retrieval(hybrid_rows, registry_ids=registry.ids, train_counts={})
    _jsonl(output / "bge_m3" / "predictions.jsonl", bge_rows)
    _jsonl(output / "char_ngram" / "predictions.jsonl", char_rows)
    _jsonl(output / "train_index" / "predictions.jsonl", index_rows)
    _jsonl(output / "hybrid" / "predictions.jsonl", hybrid_rows)
    _write_json(output / "bge_m3" / "per_class_metrics.json", bge_metrics["per_class"])
    _write_json(output / "char_ngram" / "per_class_metrics.json", char_metrics["per_class"])
    _write_json(output / "train_index" / "per_class_metrics.json", index_metrics["per_class"])
    _write_json(output / "hybrid" / "per_class_metrics.json", hybrid_metrics["per_class"])

    corpus_audit = {
        "registry_size": len(documents),
        "corpus_checksum": corpus_checksum,
        "registry_description_count": sum(item.provenance == "registry_description" for item in documents),
        "label_name_fallback_count": sum(item.provenance == "label_name_fallback" for item in documents),
        "documents": [item.__dict__ for item in documents],
    }
    data_audit = {
        "requested_splits": ["train", "val"],
        "evaluation_split": "val",
        "real_test_split_read": False,
        "metadata_fields": ["field_name"],
        "rows": len(cases),
        "unique_source_ids": len({item["source_id"] for item in cases}),
        "registry_size": len(registry.ids),
        "candidate_source_split": "train",
        "evaluation_split": "val",
        "train_rows": len(train_records),
        "train_index_unique_fields": len(candidate_index),
    }
    report = {
        **data_audit,
        "embedding_method": "bge-m3-dense-cls-normalized",
        "model_identity": str(getattr(encoder, "model_identity", "BAAI/bge-m3")),
        "embedding_dimension": int(np.asarray(corpus_vectors).shape[1]),
        "corpus_checksum": corpus_checksum,
        "candidate_source_split": "train",
        "train_index_rows": len(train_records),
        "train_index_unique_fields": len(candidate_index),
        "bge_m3": {key: value for key, value in bge_metrics.items() if key != "per_class"},
        "char_ngram": {key: value for key, value in char_metrics.items() if key != "per_class"},
        "train_index": {key: value for key, value in index_metrics.items() if key != "per_class"},
        "hybrid": {key: value for key, value in hybrid_metrics.items() if key != "per_class"},
        "hybrid_weights": {
            "lexical": float(lexical_weight),
            "dense": float(dense_weight),
            "train_index": float(index_weight),
        },
        "delta": {
            "recall_at_1": bge_metrics["recall_at_1"] - char_metrics["recall_at_1"],
            "recall_at_3": bge_metrics["recall_at_3"] - char_metrics["recall_at_3"],
            "recall_at_5": bge_metrics["recall_at_5"] - char_metrics["recall_at_5"],
            "macro_recall_at_5": bge_metrics["macro_recall_at_5"] - char_metrics["macro_recall_at_5"],
        },
        "hybrid_delta_vs_bge": {
            "recall_at_1": hybrid_metrics["recall_at_1"] - bge_metrics["recall_at_1"],
            "recall_at_3": hybrid_metrics["recall_at_3"] - bge_metrics["recall_at_3"],
            "recall_at_5": hybrid_metrics["recall_at_5"] - bge_metrics["recall_at_5"],
            "macro_recall_at_5": hybrid_metrics["macro_recall_at_5"] - bge_metrics["macro_recall_at_5"],
        },
        "runtime": {
            "bge_encoding_and_scoring_seconds": encode_elapsed,
            "char_ngram_scoring_seconds": lexical_elapsed,
            "train_index_scoring_seconds": index_elapsed,
            "total_seconds": time.monotonic() - started,
            "gpu_peak_memory_bytes": (
                int(torch.cuda.max_memory_allocated())
                if torch is not None and torch.cuda.is_available()
                else 0
            ),
        },
    }
    _write_json(output / "data_audit.json", data_audit)
    _write_json(output / "corpus_audit.json", corpus_audit)
    _write_json(output / "evaluation_report.json", report)
    _write_json(
        output / "comparison_to_char_ngram.json",
        {
            "bge_m3": report["bge_m3"],
            "char_ngram": report["char_ngram"],
            "train_index": report["train_index"],
            "hybrid": report["hybrid"],
            "delta": report["delta"],
            "hybrid_delta_vs_bge": report["hybrid_delta_vs_bge"],
            "hybrid_weights": report["hybrid_weights"],
        },
    )
    (output / "COMPLETE").write_text("complete\n", encoding="utf-8")
    return report
