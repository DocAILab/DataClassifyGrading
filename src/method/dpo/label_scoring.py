"""Deterministic semantic retrieval and memory-bounded SFT answer scoring."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
from typing import Any, Mapping, Sequence

from agent.task import LeafRegistry, TaskConfig
from agent.task.prompts import build_stage2_prompt, stage2_answer


def _features(value: str) -> Counter[str]:
    normalized = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", " ", value.lower()).strip()
    compact = normalized.replace(" ", "")
    features: Counter[str] = Counter()
    for token in normalized.split():
        features[f"token:{token}"] += 5
    for size in (1, 2, 3):
        features.update(
            f"char{size}:{compact[index:index + size]}"
            for index in range(max(0, len(compact) - size + 1))
        )
    return features


def _cosine(left: Counter[str], right: Counter[str]) -> float:
    numerator = sum(value * right.get(key, 0) for key, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    return 0.0 if not left_norm or not right_norm else numerator / (left_norm * right_norm)


def retrieve_semantic_hard_negatives(
    field_name: str,
    golden: str,
    registry: LeafRegistry,
    *,
    count: int = 4,
) -> list[str]:
    """Retrieve deterministic label/description neighbors without using val/test."""
    if golden not in registry.ids:
        raise ValueError("golden label is absent from registry")
    if count < 1 or count > len(registry.ids) - 1:
        raise ValueError("count must fit the number of wrong registry labels")
    query = _features(field_name)
    order = {label: index for index, label in enumerate(registry.ids)}
    ranked = sorted(
        (
            (
                _cosine(
                    query,
                    _features(f"{category.category_id} {category.description}"),
                ),
                category.category_id,
            )
            for category in registry.categories
            if category.category_id != golden
        ),
        key=lambda item: (-item[0], order[item[1]]),
    )
    return [label for _, label in ranked[:count]]


def completion_mean_log_probs(logits, token_ids, mask):
    """Return masked mean token log-prob without materializing log_softmax."""
    import torch

    if logits.ndim != 3 or token_ids.shape != logits.shape[:2] or mask.shape != token_ids.shape:
        raise ValueError("logits, token_ids and mask shapes are inconsistent")
    counts = mask.sum(dim=-1)
    if torch.any(counts == 0):
        raise ValueError("each completion must contain at least one token")
    selected = logits.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)
    token_log_probs = selected - torch.logsumexp(logits, dim=-1)
    return (token_log_probs * mask).sum(dim=-1) / counts


def _token_ids(tokenizer, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    ids = encoded["input_ids"]
    return ids.tolist() if hasattr(ids, "tolist") else list(ids)


def score_candidate_answers(
    model,
    tokenizer,
    prompt_messages: Sequence[Mapping[str, str]],
    answers: Mapping[str, str],
    *,
    batch_size: int = 8,
    device: str = "cuda",
) -> dict[str, float]:
    """Score assistant completions using only the final completion-token logits."""
    import torch

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not answers:
        raise ValueError("answers must not be empty")
    prompt_text = tokenizer.apply_chat_template(
        list(prompt_messages), tokenize=False, add_generation_prompt=True
    )
    prompt_length = len(_token_ids(tokenizer, prompt_text))
    items = list(answers.items())
    results: dict[str, float] = {}
    original_padding_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"
    try:
        model.eval()
        for start in range(0, len(items), batch_size):
            chunk = items[start:start + batch_size]
            full_texts = [
                tokenizer.apply_chat_template(
                    [*prompt_messages, {"role": "assistant", "content": answer}],
                    tokenize=False,
                    add_generation_prompt=False,
                )
                for _, answer in chunk
            ]
            full_lengths = [len(_token_ids(tokenizer, text)) for text in full_texts]
            completion_lengths = [length - prompt_length for length in full_lengths]
            if any(length <= 0 for length in completion_lengths):
                raise ValueError("chat template produced an empty completion")
            encoded = tokenizer(
                full_texts,
                padding=True,
                return_tensors="pt",
                add_special_tokens=False,
            )
            encoded = {
                key: value.to(device) if hasattr(value, "to") else value
                for key, value in encoded.items()
            }
            keep = max(completion_lengths) + 1
            with torch.inference_mode():
                try:
                    output = model(**encoded, logits_to_keep=keep)
                except TypeError as exc:
                    raise RuntimeError(
                        "model must support logits_to_keep for bounded-memory scoring"
                    ) from exc
            logits = output.logits
            width = logits.shape[1]
            target_ids = torch.zeros(
                (len(chunk), width - 1), dtype=torch.long, device=logits.device
            )
            mask = torch.zeros_like(target_ids, dtype=torch.bool)
            full_input_ids = encoded["input_ids"]
            for row, completion_length in enumerate(completion_lengths):
                target_ids[row, -completion_length:] = full_input_ids[row, -completion_length:]
                mask[row, -completion_length:] = True
            means = completion_mean_log_probs(logits[:, :-1, :], target_ids, mask)
            for (label, _), value in zip(chunk, means.tolist(), strict=True):
                results[label] = float(value)
    finally:
        tokenizer.padding_side = original_padding_side
    return results


def load_completed_score_rows(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load verified JSONL rows used to skip completed source IDs on resume."""
    source = Path(path)
    if not source.is_file():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            source_id = str(value.get("source_id", "")).strip()
            if not source_id:
                raise ValueError(f"score row {line_number} has no source_id")
            if source_id in rows:
                raise ValueError(f"duplicate score row for source_id {source_id}")
            if not isinstance(value.get("scores"), Mapping):
                raise ValueError(f"score row {line_number} has invalid scores")
            rows[source_id] = value
    return rows


def mine_score_rows(
    records: Sequence[Mapping[str, Any]],
    registry: LeafRegistry,
    output_path: str | Path,
    *,
    score_fn,
    model_identity: str,
    seed: int = 42,
) -> dict[str, Any]:
    """Mine five-way SFT scores with per-source durable JSONL resume."""
    if not model_identity.strip():
        raise ValueError("model_identity must be non-empty")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = load_completed_score_rows(output)
    for source_id, value in completed.items():
        identity = str(value.get("model_identity", "")).strip()
        if identity and identity != model_identity:
            raise ValueError(f"existing score row uses another model: {source_id}")
    existing_count = len(completed)
    new_count = 0
    seen: set[str] = set()
    config = TaskConfig(("field_name",), "field_name_level4")
    for record in records:
        source_id = str(record.get("id", "")).strip()
        if not source_id or source_id in seen:
            raise ValueError("records require unique non-empty source IDs")
        seen.add(source_id)
        if source_id in completed:
            continue
        metadata = record.get("metadata")
        classification = record.get("classification")
        if not isinstance(metadata, Mapping) or not isinstance(classification, Mapping):
            raise ValueError(f"record {source_id} has invalid metadata or classification")
        field_name = str(metadata.get("field_name", "") or "")
        golden = str(classification.get("level_4", "")).strip()
        negatives = retrieve_semantic_hard_negatives(field_name, golden, registry, count=4)
        candidates = [golden, *negatives]
        digest = hashlib.sha256(f"{seed}\0{source_id}".encode("utf-8")).digest()
        random.Random(int.from_bytes(digest[:16], "big")).shuffle(candidates)
        prompt = build_stage2_prompt(
            {"field_name": field_name}, candidates, registry, config
        )
        prompt_messages = [
            {"role": "system", "content": prompt.system},
            {"role": "user", "content": prompt.user},
        ]
        answers = {label: stage2_answer(label, candidates) for label in candidates}
        scores = score_fn(prompt_messages, answers)
        if set(scores) != set(candidates):
            raise ValueError(f"scorer returned incomplete candidate scores: {source_id}")
        row = {
            "source_id": source_id,
            "scores": {label: float(scores[label]) for label in candidates},
            "retrieved_negatives": negatives,
            "scoring_candidates": candidates,
            "retrieval_policy": "field_registry_char_ngram_v1",
            "model_identity": model_identity,
            "seed": seed,
            "metadata_fields": ["field_name"],
            "real_test_split_read": False,
        }
        with output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        completed[source_id] = row
        new_count += 1
    return {
        "requested_splits": ["train"],
        "real_test_split_read": False,
        "metadata_fields": ["field_name"],
        "model_identity": model_identity,
        "existing_rows": existing_count,
        "new_rows": new_count,
        "total_rows": len(completed),
    }
