"""Train-only hard-negative preference rows for Stage 2 DPO."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Mapping

from agent.task import LeafRegistry, TaskConfig
from agent.task.prompts import build_stage2_prompt, stage2_answer
from method.sft.dataset import load_json_records


def _registry(value: LeafRegistry | str | Path) -> LeafRegistry:
    return value if isinstance(value, LeafRegistry) else LeafRegistry.from_path(value)


def _config(value: TaskConfig | str | Path) -> TaskConfig:
    return value if isinstance(value, TaskConfig) else TaskConfig.from_path(value)


def load_train_records(input_dir: str | Path) -> list[dict[str, Any]]:
    """Load only train.json; validation and test paths are never resolved."""
    return load_json_records(Path(input_dir) / "train.json")


def select_hard_candidates(
    ground_truth: str,
    scores: Mapping[str, float],
    registry: LeafRegistry,
    *,
    source_id: str,
    seed: int = 42,
) -> tuple[list[str], str]:
    """Select the four highest-scoring wrong labels and shuffle with golden."""
    stable_id = source_id.strip()
    if not stable_id:
        raise ValueError("source_id must be non-empty")
    if ground_truth not in registry.ids:
        raise ValueError("ground truth is OOV")
    unknown = set(scores) - set(registry.ids)
    if unknown:
        raise ValueError(f"score map contains OOV labels: {sorted(unknown)}")
    available = [label for label in registry.ids if label in scores]
    if ground_truth not in scores or len(available) < 5:
        raise ValueError("score map must contain the golden label and at least five labels")
    numeric: dict[str, float] = {}
    for label in available:
        value = float(scores[label])
        if not math.isfinite(value):
            raise ValueError(f"score must be finite: {label}")
        numeric[label] = value
    registry_order = {label: index for index, label in enumerate(registry.ids)}
    wrong = sorted(
        (label for label in available if label != ground_truth),
        key=lambda label: (-numeric[label], registry_order[label]),
    )
    if len(wrong) < 4:
        raise ValueError("score map must contain at least four wrong labels")
    hard_negatives = wrong[:4]
    candidates = [ground_truth, *hard_negatives]
    digest = hashlib.sha256(f"{seed}\0{stable_id}".encode("utf-8")).digest()
    random.Random(int.from_bytes(digest[:16], "big")).shuffle(candidates)
    return candidates, hard_negatives[0]


def _ground_truth(record: Mapping[str, Any]) -> str:
    classification = record.get("classification")
    if not isinstance(classification, Mapping):
        raise ValueError("record classification must be an object")
    value = str(classification.get("level_4", "")).strip()
    if not value:
        raise ValueError("record must have classification.level_4")
    return value


def build_preference_row(
    record: Mapping[str, Any],
    scores: Mapping[str, float],
    registry: LeafRegistry,
    config: TaskConfig,
    *,
    seed: int = 42,
) -> dict[str, Any]:
    """Build one conversational DPO row without leaking hidden metadata."""
    if config.metadata_fields != ("field_name",):
        raise ValueError("DPO contract requires metadata_fields=[field_name]")
    source_id = str(record.get("id", "")).strip()
    if not source_id:
        raise ValueError("record must have a stable source_id")
    metadata = record.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("record metadata must be an object")
    visible_metadata = {"field_name": metadata.get("field_name", "") or ""}
    golden = _ground_truth(record)
    candidates, rejected_label = select_hard_candidates(
        golden, scores, registry, source_id=source_id, seed=seed
    )
    prompt = build_stage2_prompt(visible_metadata, candidates, registry, config)
    prompt_messages = [
        {"role": "system", "content": prompt.system},
        {"role": "user", "content": prompt.user},
    ]
    serialized_prompt = json.dumps(
        prompt_messages, ensure_ascii=False, separators=(",", ":")
    )
    return {
        "source_id": source_id,
        "prompt": prompt_messages,
        "prompt_sha256": hashlib.sha256(serialized_prompt.encode("utf-8")).hexdigest(),
        "chosen": [{"role": "assistant", "content": stage2_answer(golden, candidates)}],
        "rejected": [
            {"role": "assistant", "content": stage2_answer(rejected_label, candidates)}
        ],
        "candidates": candidates,
        "ground_truth": golden,
        "rejected_label": rejected_label,
        "hard_negative_scores": json.dumps(
            {label: float(scores[label]) for label in candidates if label != golden},
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ),
        "metadata": visible_metadata,
        "seed": seed,
        "golden_position": candidates.index(golden),
    }


def _load_scores(path: str | Path) -> tuple[dict[str, dict[str, float]], list[str]]:
    by_source: dict[str, dict[str, float]] = {}
    identities: set[str] = set()
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            source_id = str(value.get("source_id", "")).strip()
            if not source_id:
                raise ValueError(f"score row {line_number} has no source_id")
            if source_id in by_source:
                raise ValueError(f"duplicate score row for source_id {source_id}")
            scores = value.get("scores")
            if not isinstance(scores, Mapping):
                raise ValueError(f"score row {line_number} has invalid scores")
            by_source[source_id] = {str(key): float(item) for key, item in scores.items()}
            identity = str(value.get("model_identity", "")).strip()
            if identity:
                identities.add(identity)
    return by_source, sorted(identities)


def export_preferences(
    input_dir: str | Path,
    output_dir: str | Path,
    registry: LeafRegistry | str | Path,
    task_config: TaskConfig | str | Path,
    score_path: str | Path,
    *,
    seed: int = 42,
) -> dict[str, Any]:
    """Export train-only conversational DPO rows and their audit report."""
    leaf_registry = _registry(registry)
    config = _config(task_config)
    records = load_train_records(input_dir)
    scores_by_source, identities = _load_scores(score_path)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        source_id = str(record.get("id", "")).strip()
        if source_id in seen:
            raise ValueError(f"duplicate train source_id: {source_id}")
        seen.add(source_id)
        if source_id not in scores_by_source:
            raise ValueError(f"missing score row for source_id {source_id}")
        rows.append(
            build_preference_row(
                record, scores_by_source[source_id], leaf_registry, config, seed=seed
            )
        )
    extras = set(scores_by_source) - seen
    if extras:
        raise ValueError(f"score rows have no matching train record: {sorted(extras)[:5]}")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("preference export requires pyarrow") from exc
    pq.write_table(pa.Table.from_pylist(rows), output / "preferences.parquet")
    report = {
        "format": "trl_conversational_preference_parquet",
        "requested_splits": ["train"],
        "real_test_split_read": False,
        "metadata_fields": list(config.metadata_fields),
        "supervision_target": "classification.level_4",
        "candidate_policy": "sft_score_hard_negative_shuffled",
        "candidate_seed": seed,
        "rows": len(rows),
        "candidate_duplicate_rows": sum(
            len(row["candidates"]) != len(set(row["candidates"])) for row in rows
        ),
        "candidate_oov_rows": sum(
            any(label not in leaf_registry.ids for label in row["candidates"])
            for row in rows
        ),
        "score_model_identities": identities,
    }
    (output / "preference_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report
