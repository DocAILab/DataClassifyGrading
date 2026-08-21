"""Exporter and validator for the VeRL SFT messages parquet baseline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
from typing import Any, Mapping

from agent.evaluation import evaluate_stage1, evaluate_stage2
from agent.task.contracts import LeafRegistry, TaskConfig
from agent.task.prompts import (
    build_stage1_prompt,
    build_stage2_prompt,
    stage1_answer,
    stage2_answer,
)

SPLITS = ("train", "val", "test")
PRODUCTION_SPLITS = ("train", "val")
CANDIDATE_POLICY_VERSION = "random_shuffled_v1"


def _registry(value: LeafRegistry | str | Path) -> LeafRegistry:
    return value if isinstance(value, LeafRegistry) else LeafRegistry.from_path(value)


def _config(value: TaskConfig | str | Path) -> TaskConfig:
    return value if isinstance(value, TaskConfig) else TaskConfig.from_path(value)


def load_json_records(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"input split file not found: {source}")
    with source.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, list):
        raise ValueError(f"input split must be a JSON array: {source}")
    if not all(isinstance(item, dict) for item in value):
        raise ValueError(f"input split items must be objects: {source}")
    return value


def load_splits(
    input_dir: str | Path,
    splits: tuple[str, ...] | list[str] = PRODUCTION_SPLITS,
) -> dict[str, list[dict[str, Any]]]:
    """Load explicit production splits without resolving or opening test data."""
    requested = tuple(splits)
    if not requested:
        raise ValueError("at least one split is required")
    if any(split not in PRODUCTION_SPLITS for split in requested):
        raise ValueError("only train and val splits are permitted; test is forbidden")
    if len(set(requested)) != len(requested):
        raise ValueError("split names must be unique")
    root = Path(input_dir)
    return {split: load_json_records(root / f"{split}.json") for split in requested}


def build_candidates(ground_truth: str, registry: LeafRegistry) -> list[str]:
    """Minimal deterministic fixture policy: GT followed by first four non-GT IDs."""
    if ground_truth not in registry.ids:
        raise ValueError(f"ground-truth category_id is absent from leaf registry: {ground_truth}")
    return [ground_truth] + [category_id for category_id in registry.ids if category_id != ground_truth][:4]


def build_random_shuffled_candidates(
    ground_truth: str,
    registry: LeafRegistry,
    *,
    source_id: str,
    seed: int = 42,
) -> list[str]:
    """Return one golden label and four stable random negatives in shuffled order."""
    stable_id = source_id.strip()
    if not stable_id:
        raise ValueError("source_id must be non-empty")
    if ground_truth not in registry.ids:
        raise ValueError("ground truth must belong to the leaf registry")
    negatives = [value for value in registry.ids if value != ground_truth]
    if len(negatives) < 4:
        raise ValueError("at least five unique registry labels are required")
    digest = hashlib.sha256(f"{seed}\0{stable_id}".encode("utf-8")).digest()
    rng = random.Random(int.from_bytes(digest[:16], "big"))
    candidates = [ground_truth, *rng.sample(negatives, 4)]
    rng.shuffle(candidates)
    return candidates


def _label(item: Mapping[str, Any], index: int, source: Path) -> str | None:
    classification = item.get("classification")
    if not isinstance(classification, Mapping):
        raise ValueError(f"item {index} in {source} has invalid classification")
    ground_truth = str(classification.get("level_4", "")).strip()
    status = item.get("label_status")
    if status == "unlabeled" and ground_truth:
        raise ValueError(f"item {index} in {source} is unlabeled but has level_4")
    if status == "labeled" and not ground_truth:
        raise ValueError(f"item {index} in {source} is labeled but has no level_4")
    return ground_truth or None


def _row(
    item: Mapping[str, Any],
    source: Path,
    index: int,
    registry: LeafRegistry,
    config: TaskConfig,
    stage: str,
    *,
    candidate_policy: str,
    candidate_seed: int,
) -> dict[str, Any]:
    metadata = item.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError(f"item {index} in {source} has invalid metadata")
    ground_truth = _label(item, index, source)
    if ground_truth is None:
        raise ValueError("internal error: unlabeled record passed to row builder")
    visible_metadata = {
        field: "" if metadata.get(field) is None else metadata.get(field, "")
        for field in config.metadata_fields
    }
    source_id = str(item.get("id", "")).strip()
    if not source_id:
        raise ValueError(f"item {index} in {source} has no stable id")
    if candidate_policy == "fixed-registry":
        candidates = build_candidates(ground_truth, registry)
        candidate_policy_version = "fixed_registry_v1"
    elif candidate_policy == "random-shuffled":
        candidates = build_random_shuffled_candidates(
            ground_truth,
            registry,
            source_id=source_id,
            seed=candidate_seed,
        )
        candidate_policy_version = CANDIDATE_POLICY_VERSION
    else:
        raise ValueError(
            "candidate_policy must be fixed-registry or random-shuffled"
        )
    if stage == "stage1":
        prompt = build_stage1_prompt(visible_metadata, registry, config)
        answer = stage1_answer(candidates)
    else:
        prompt = build_stage2_prompt(visible_metadata, candidates, registry, config)
        answer = stage2_answer(ground_truth, candidates)
    return {
        "messages": [
            {"role": "system", "content": prompt.system},
            {"role": "user", "content": prompt.user},
            {"role": "assistant", "content": answer},
        ],
        "stage": stage,
        "source_id": source_id,
        "metadata": visible_metadata,
        "ground_truth": ground_truth,
        "candidates": candidates,
        "candidate_policy": candidate_policy,
        "candidate_policy_version": candidate_policy_version,
        "candidate_seed": candidate_seed,
        "golden_position": candidates.index(ground_truth),
    }


def _write_parquet(rows: list[dict[str, Any]], path: Path) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - dependency install issue
        raise RuntimeError("SFT export requires pyarrow") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def export_sft_dataset(
    input_dir: str | Path,
    output_dir: str | Path,
    registry: LeafRegistry | str | Path,
    task_config: TaskConfig | str | Path,
    splits: tuple[str, ...] | list[str] = PRODUCTION_SPLITS,
    *,
    candidate_policy: str = "random-shuffled",
    candidate_seed: int = 42,
) -> dict[str, Any]:
    """Export each labeled JSON record as one stage1 and one stage2 SFT row."""
    leaf_registry = _registry(registry)
    config = _config(task_config)
    if candidate_policy not in {"fixed-registry", "random-shuffled"}:
        raise ValueError(
            "candidate_policy must be fixed-registry or random-shuffled"
        )
    input_root, output_root = Path(input_dir), Path(output_dir)
    candidate_policy_version = (
        CANDIDATE_POLICY_VERSION
        if candidate_policy == "random-shuffled"
        else "fixed_registry_v1"
    )
    report: dict[str, Any] = {
        "format": "verl_sft_messages_parquet",
        "candidate_policy": candidate_policy,
        "candidate_policy_version": candidate_policy_version,
        "candidate_seed": candidate_seed,
        "golden_position_histogram": {str(index): 0 for index in range(5)},
        "candidate_duplicate_rows": 0,
        "candidate_oov_rows": 0,
        "metadata_fields": list(config.metadata_fields),
        "supervision_target": "classification.level_4",
        "external_corpus": "leaf_registry_descriptions",
        "task_name": config.task_name,
        "registry_size": len(leaf_registry.categories),
        "splits": {},
        "requested_splits": list(splits),
        "real_test_split_read": False,
    }
    for split, records in load_splits(input_root, splits).items():
        source = input_root / f"{split}.json"
        rows: list[dict[str, Any]] = []
        skipped = 0
        for index, item in enumerate(records):
            ground_truth = _label(item, index, source)
            if ground_truth is None:
                skipped += 1
                continue
            rows.extend(
                _row(
                    item,
                    source,
                    index,
                    leaf_registry,
                    config,
                    stage,
                    candidate_policy=candidate_policy,
                    candidate_seed=candidate_seed,
                )
                for stage in ("stage1", "stage2")
            )
            candidates = rows[-1]["candidates"]
            report["golden_position_histogram"][
                str(candidates.index(ground_truth))
            ] += 1
            if len(candidates) != len(set(candidates)):
                report["candidate_duplicate_rows"] += 1
            if any(candidate not in leaf_registry.ids for candidate in candidates):
                report["candidate_oov_rows"] += 1
        if not rows:
            raise ValueError(f"no labeled records available in {source}")
        destination = output_root / f"{split}.parquet"
        _write_parquet(rows, destination)
        report["splits"][split] = {
            "input_records": len(records),
            "exported_records": len(rows),
            "skipped_unlabeled": skipped,
            "output_file": str(destination),
        }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "export_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def _validate_row(
    row: Mapping[str, Any], registry: LeafRegistry, config: TaskConfig
) -> list[str]:
    errors: list[str] = []
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) != 3:
        return ["messages must contain system, user, assistant"]
    roles = [message.get("role") if isinstance(message, Mapping) else None for message in messages]
    if roles != ["system", "user", "assistant"]:
        errors.append("messages roles must be system, user, assistant")
    contents = [
        message.get("content") if isinstance(message, Mapping) else None
        for message in messages
    ]
    if not all(isinstance(content, str) for content in contents):
        errors.append("every message content must be a string")
    elif any(
        token in "".join(contents)
        for token in ("<|im_start|>", "<|im_end|>")
    ):
        errors.append("messages must not contain serialized chat-template tokens")
    source_id = row.get("source_id")
    if not isinstance(source_id, str) or not source_id.strip():
        errors.append("source_id must be a non-empty string")
    assistant = messages[-1].get("content") if isinstance(messages[-1], Mapping) else None
    if not isinstance(assistant, str):
        return errors + ["assistant content must be a JSON string"]
    stage = row.get("stage")
    ground_truth = row.get("ground_truth")
    candidates = row.get("candidates")
    candidates_are_valid = (
        isinstance(candidates, list)
        and all(isinstance(candidate, str) for candidate in candidates)
        and len(candidates) == 5
        and len(set(candidates)) == 5
    )
    if not candidates_are_valid:
        errors.append("candidates must contain exactly 5 unique string IDs")
    elif any(candidate not in registry.ids for candidate in candidates):
        errors.append("candidates contain an ID absent from registry")
    if not isinstance(ground_truth, str) or ground_truth not in registry.ids:
        errors.append("ground_truth must be an ID from the leaf registry")
    candidates_belong_to_registry = candidates_are_valid and all(
        candidate in registry.ids for candidate in candidates
    )
    ground_truth_is_valid = (
        isinstance(ground_truth, str) and ground_truth in registry.ids
    )
    if stage == "stage1" and ground_truth_is_valid:
        evaluation = evaluate_stage1(
            assistant,
            ground_truth=ground_truth,
            registry=registry,
        )
        errors.extend(f"stage1 evaluation: {error}" for error in evaluation.errors)
        if evaluation.prediction is None or list(evaluation.prediction) != candidates:
            errors.append("stage1 answer must exactly match the five candidates")
    elif stage == "stage2" and candidates_belong_to_registry and ground_truth_is_valid:
        evaluation = evaluate_stage2(
            assistant,
            ground_truth=ground_truth,
            candidates=tuple(candidates),
            registry=registry,
        )
        errors.extend(f"stage2 evaluation: {error}" for error in evaluation.errors)
        if evaluation.contract_valid and not evaluation.correct:
            errors.append("stage2 answer must equal ground_truth")
    elif stage not in {"stage1", "stage2"}:
        errors.append("stage must be stage1 or stage2")

    visible_metadata = row.get("metadata")
    expected_fields = set(config.metadata_fields)
    if not isinstance(visible_metadata, Mapping) or set(visible_metadata) != expected_fields:
        errors.append("metadata must exactly match task config metadata_fields")
    elif all(isinstance(content, str) for content in contents):
        expected_prompt = None
        if stage == "stage1":
            expected_prompt = build_stage1_prompt(visible_metadata, registry, config)
        elif (
            stage == "stage2"
            and candidates_are_valid
            and all(candidate in registry.ids for candidate in candidates)
        ):
            expected_prompt = build_stage2_prompt(
                visible_metadata, candidates, registry, config
            )
        if expected_prompt is not None and contents[:2] != [
            expected_prompt.system,
            expected_prompt.user,
        ]:
            errors.append("system/user prompt does not match registry and task contract")

    if isinstance(ground_truth, str) and ground_truth in registry.ids:
        candidate_policy = row.get("candidate_policy", "fixed-registry")
        if candidate_policy == "fixed-registry":
            expected = build_candidates(ground_truth, registry)
        elif candidate_policy == "random-shuffled":
            candidate_seed = row.get("candidate_seed")
            if not isinstance(candidate_seed, int) or isinstance(candidate_seed, bool):
                errors.append("random-shuffled candidates require an integer seed")
                return errors
            if not isinstance(source_id, str) or not source_id.strip():
                return errors
            expected = build_random_shuffled_candidates(
                ground_truth,
                registry,
                source_id=source_id,
                seed=candidate_seed,
            )
        else:
            errors.append("candidate_policy must be fixed-registry or random-shuffled")
            return errors
        if candidates != expected:
            errors.append("candidates do not match their declared deterministic policy")
    return errors


def _validate_stage_pairs(rows: list[Mapping[str, Any]]) -> list[str]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        source_id = row.get("source_id")
        if isinstance(source_id, str) and source_id.strip():
            grouped.setdefault(source_id, []).append(row)

    errors: list[str] = []
    for source_id, source_rows in grouped.items():
        stages = [row.get("stage") for row in source_rows]
        if len(source_rows) != 2 or set(stages) != {"stage1", "stage2"}:
            errors.append(
                f"source_id {source_id!r} must have exactly one stage1 and stage2 row"
            )
            continue
        ground_truths = {row.get("ground_truth") for row in source_rows}
        candidate_lists = {
            json.dumps(row.get("candidates"), ensure_ascii=False, sort_keys=True)
            for row in source_rows
        }
        if len(ground_truths) != 1 or len(candidate_lists) != 1:
            errors.append(
                f"source_id {source_id!r} must share ground_truth and candidates across stages"
            )
    return errors


def validate_sft_dataset(
    dataset_dir: str | Path,
    registry: LeafRegistry | str | Path,
    task_config: TaskConfig | str | Path,
    splits: tuple[str, ...] | list[str] = PRODUCTION_SPLITS,
) -> dict[str, Any]:
    """Return a structured report; malformed rows are reported, not silently accepted."""
    leaf_registry = _registry(registry)
    config = _config(task_config)
    root = Path(dataset_dir)
    requested = tuple(splits)
    if not requested or any(split not in PRODUCTION_SPLITS for split in requested):
        raise ValueError("splits must be a non-empty subset of: train, val")
    if len(set(requested)) != len(requested):
        raise ValueError("split names must be unique")
    report: dict[str, Any] = {
        "format": "verl_sft_messages_parquet",
        "valid": True,
        "metadata_fields": list(config.metadata_fields),
        "task_name": config.task_name,
        "registry_size": len(leaf_registry.categories),
        "splits": {},
        "cross_split_errors": [],
        "requested_splits": list(requested),
        "real_test_split_read": False,
    }
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("SFT validation requires pyarrow") from exc
    source_ids_by_split: dict[str, set[str]] = {}
    for split in requested:
        path = root / f"{split}.parquet"
        details = {"rows": 0, "errors": []}
        rows: list[Mapping[str, Any]] = []
        if not path.is_file():
            details["errors"].append(f"missing parquet file: {path}")
        else:
            try:
                rows = pq.read_table(path).to_pylist()
                details["rows"] = len(rows)
                if not rows:
                    details["errors"].append("parquet split must contain at least one row")
                for index, row in enumerate(rows):
                    for error in _validate_row(row, leaf_registry, config):
                        details["errors"].append(f"row {index}: {error}")
                details["errors"].extend(_validate_stage_pairs(rows))
            except Exception as exc:  # malformed parquet is a validation failure
                details["errors"].append(f"cannot read parquet: {exc}")
        source_ids_by_split[split] = {
            source_id
            for row in rows
            if isinstance((source_id := row.get("source_id")), str)
            and source_id.strip()
        }
        report["splits"][split] = details
        if details["errors"]:
            report["valid"] = False

    for left_index, left in enumerate(requested):
        for right in requested[left_index + 1 :]:
            overlap = source_ids_by_split[left] & source_ids_by_split[right]
            if overlap:
                examples = ", ".join(sorted(overlap)[:5])
                report["cross_split_errors"].append(
                    f"source_id overlap between {left} and {right}: {examples}"
                )
    if report["cross_split_errors"]:
        report["valid"] = False
    return report
