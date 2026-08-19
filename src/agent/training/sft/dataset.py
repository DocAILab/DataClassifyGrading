"""Exporter and validator for the VeRL SFT messages parquet baseline.

Production labels come EXCLUSIVELY from the canonical dataset contract:
records are read from data/canonical/<dataset>/all.json, only
resolution_status == "resolved" records with a target whose category_id
belongs to the LeafRegistry enter training, and the ground truth is always
record["target"]["category_id"]. classification.level_1..level_4 stay
provenance only — there is deliberately NO fallback to them as labels.

Split boundaries (train/val/test) are taken from the original split JSON
files by record id; a split record whose id is missing from the canonical
file is a data-consistency error (fail-fast), while resolved/unresolved
filtering happens on the canonical status.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from agent.evaluation import evaluate_stage1_choices, evaluate_stage2_choices
from agent.task.contracts import CorpusCategory, LeafRegistry, TaskConfig
from agent.task.prompt_choices import PromptChoiceRegistry
from agent.task.prompts import (
    build_stage1_prompt,
    build_stage2_prompt,
    stage1_answer,
    stage2_answer,
)
from agent.training.common import (
    build_candidates,
    canonical_target as _canonical_target,
    require_corpus as _require_corpus,
    require_corpus_covers_registry as _require_corpus_covers_registry,
)

SPLITS = ("train", "val", "test")


def _registry(value: LeafRegistry | str | Path) -> LeafRegistry:
    return value if isinstance(value, LeafRegistry) else LeafRegistry.from_path(value)


def _config(value: TaskConfig | str | Path) -> TaskConfig:
    return value if isinstance(value, TaskConfig) else TaskConfig.from_path(value)


def _corpus_map(corpus: Mapping[str, CorpusCategory]) -> Mapping[str, CorpusCategory]:
    return _require_corpus(corpus)


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


def _row(
    item: Mapping[str, Any],
    source: Path,
    index: int,
    registry: LeafRegistry,
    config: TaskConfig,
    stage: str,
    corpus: Mapping[str, CorpusCategory],
) -> dict[str, Any]:
    metadata = item.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError(f"item {index} in {source} has invalid metadata")
    ground_truth = _canonical_target(item, index, source, registry)
    if ground_truth is None:
        raise ValueError("internal error: unresolved record passed to row builder")
    visible_metadata = {
        field: "" if metadata.get(field) is None else metadata.get(field, "")
        for field in config.metadata_fields
    }
    source_id = str(item.get("id", "")).strip()
    if not source_id:
        raise ValueError(f"item {index} in {source} has no stable id")
    candidates = build_candidates(ground_truth, registry, source_id=source_id)
    choices = PromptChoiceRegistry.from_registry(registry)
    if stage == "stage1":
        prompt = build_stage1_prompt(visible_metadata, registry, config, choices=choices)
        answer = stage1_answer(candidates, choices=choices)
    else:
        prompt = build_stage2_prompt(
            visible_metadata,
            candidates,
            registry,
            config,
            corpus=corpus or None,
            choices=choices,
        )
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
    canonical_file: str | Path,
    split_dir: str | Path,
    output_dir: str | Path,
    registry: LeafRegistry | str | Path,
    task_config: TaskConfig | str | Path,
    corpus: Mapping[str, CorpusCategory],
) -> dict[str, Any]:
    """Export canonical records to one stage1 + one stage2 SFT row per split.

    Labels come exclusively from target.category_id (canonical contract);
    split boundaries follow the original split JSON files by record id.
    corpus is REQUIRED for the production Stage 2 path: candidates are
    resolved by category_id against the canonical corpus and there is no
    registry fallback in the exporter.
    """
    leaf_registry = _registry(registry)
    config = _config(task_config)
    canonical_path, split_root, output_root = (
        Path(canonical_file),
        Path(split_dir),
        Path(output_dir),
    )
    if not canonical_path.is_file():
        raise FileNotFoundError(f"canonical dataset not found: {canonical_path}")
    canonical_records = load_json_records(canonical_path)
    canonical_by_id: dict[str, dict[str, Any]] = {}
    canonical_resolved = 0
    idless_non_resolved = 0
    for index, item in enumerate(canonical_records):
        item_id = str(item.get("id", "") or "").strip()
        status = str(item.get("resolution_status", "") or "").strip()
        if status == "resolved":
            canonical_resolved += 1
            # every resolved record must satisfy the canonical contract
            # (target present, category_id in registry, leaf_name matches),
            # regardless of whether it belongs to any train/val/test split —
            # splits decide training membership, not contract validation
            _canonical_target(item, index, canonical_path, leaf_registry)
            if not item_id:
                raise ValueError(
                    f"resolved canonical record without id: {canonical_path}"
                )
        elif not item_id:
            # stage-3B contract allows id-less audit records for non-resolved
            # outcomes (e.g. {"record": ..., "resolution_status":
            # "invalid_record"}); they never enter training and are ignored
            # by the split join, but must not fail the export
            idless_non_resolved += 1
            continue
        if item_id in canonical_by_id:
            raise ValueError(f"duplicate id in canonical dataset: {item_id}")
        canonical_by_id[item_id] = item
    corpus_map = _corpus_map(corpus)
    _require_corpus_covers_registry(corpus_map, leaf_registry)

    report: dict[str, Any] = {
        "format": "verl_sft_messages_parquet",
        "label_source": "canonical target.category_id (resolution_status == resolved)",
        "prompt_identity": (
            "choice ids in messages; ground_truth/candidates stay canonical "
            "category_id (PromptChoiceRegistry, display names = shortest "
            "unique path suffix)"
        ),
        "candidate_policy": (
            "baseline fixture: (ground_truth + first four registry negatives) "
            "permuted deterministically by stable source_id (not position-fixed, "
            "not the production retrieval policy)"
        ),
        "metadata_fields": list(config.metadata_fields),
        "task_name": config.task_name,
        "registry_size": len(leaf_registry.categories),
        "corpus_size": len(corpus_map),
        "canonical_resolved": canonical_resolved,
        "idless_non_resolved_records": idless_non_resolved,
        "splits": {},
    }
    trainable_ids: set[str] = set()
    for split in SPLITS:
        source = split_root / f"{split}.json"
        split_records = load_json_records(source)
        rows: list[dict[str, Any]] = []
        skipped_unresolved = 0
        for index, item in enumerate(split_records):
            item_id = str(item.get("id", "") or "").strip()
            if not item_id:
                raise ValueError(f"split item {index} in {source} has no id")
            canonical_item = canonical_by_id.get(item_id)
            if canonical_item is None:
                raise ValueError(
                    f"split item id {item_id!r} ({source}) is absent from the "
                    f"canonical dataset {canonical_path}"
                )
            ground_truth = _canonical_target(canonical_item, index, source, leaf_registry)
            if ground_truth is None:
                skipped_unresolved += 1
                continue
            trainable_ids.add(item_id)
            rows.extend(
                _row(canonical_item, source, index, leaf_registry, config, stage, corpus_map)
                for stage in ("stage1", "stage2")
            )
        if not rows:
            raise ValueError(f"no resolved records available in {source}")
        destination = output_root / f"{split}.parquet"
        _write_parquet(rows, destination)
        report["splits"][split] = {
            "split_records": len(split_records),
            "exported_records": len(rows),
            "skipped_not_resolved": skipped_unresolved,
            "output_file": str(destination),
        }
    resolved_outside_split_ids = sorted(
        record_id
        for record_id in canonical_by_id
        if record_id not in trainable_ids
        and str(canonical_by_id[record_id].get("resolution_status", "")).strip()
        == "resolved"
    )
    report["trainable_resolved"] = len(trainable_ids)
    report["resolved_outside_splits"] = len(resolved_outside_split_ids)
    report["resolved_outside_split_ids"] = resolved_outside_split_ids
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "export_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def _validate_row(
    row: Mapping[str, Any],
    registry: LeafRegistry,
    config: TaskConfig,
    corpus: Mapping[str, CorpusCategory],
) -> list[str]:
    errors: list[str] = []
    choices = PromptChoiceRegistry.from_registry(registry)
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
        evaluation = evaluate_stage1_choices(
            assistant,
            ground_truth=ground_truth,
            registry=registry,
            choices=choices,
        )
        errors.extend(f"stage1 evaluation: {error}" for error in evaluation.errors)
        if evaluation.prediction is None or list(evaluation.prediction) != candidates:
            errors.append("stage1 answer must exactly match the five candidates")
    elif stage == "stage2" and candidates_belong_to_registry and ground_truth_is_valid:
        evaluation = evaluate_stage2_choices(
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
            expected_prompt = build_stage1_prompt(
                visible_metadata, registry, config, choices=choices
            )
        elif (
            stage == "stage2"
            and candidates_are_valid
            and all(candidate in registry.ids for candidate in candidates)
        ):
            expected_prompt = build_stage2_prompt(
                visible_metadata,
                candidates,
                registry,
                config,
                corpus=corpus or None,
                choices=choices,
            )
        if expected_prompt is not None and contents[:2] != [
            expected_prompt.system,
            expected_prompt.user,
        ]:
            errors.append("system/user prompt does not match registry and task contract")

    if (
        isinstance(ground_truth, str)
        and ground_truth in registry.ids
        and isinstance(source_id, str)
        and source_id.strip()
    ):
        expected = build_candidates(ground_truth, registry, source_id=source_id)
        if candidates != expected:
            errors.append("candidates do not follow the deterministic source-seeded bundle order")
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
    corpus: Mapping[str, CorpusCategory],
) -> dict[str, Any]:
    """Return a structured report; malformed rows are reported, not silently accepted.

    corpus is REQUIRED: production validation rebuilds the Stage 2 prompt
    from the canonical corpus (no registry fallback in the validator).
    """
    leaf_registry = _registry(registry)
    config = _config(task_config)
    corpus_map = _corpus_map(corpus)
    _require_corpus_covers_registry(corpus_map, leaf_registry)
    root = Path(dataset_dir)
    report: dict[str, Any] = {
        "format": "verl_sft_messages_parquet",
        "valid": True,
        "metadata_fields": list(config.metadata_fields),
        "task_name": config.task_name,
        "registry_size": len(leaf_registry.categories),
        "corpus_size": len(corpus_map),
        "splits": {},
        "cross_split_errors": [],
    }
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("SFT validation requires pyarrow") from exc
    source_ids_by_split: dict[str, set[str]] = {}
    for split in SPLITS:
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
                    for error in _validate_row(row, leaf_registry, config, corpus_map):
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

    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1 :]:
            overlap = source_ids_by_split[left] & source_ids_by_split[right]
            if overlap:
                examples = ", ".join(sorted(overlap)[:5])
                report["cross_split_errors"].append(
                    f"source_id overlap between {left} and {right}: {examples}"
                )
    if report["cross_split_errors"]:
        report["valid"] = False
    return report
