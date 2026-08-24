"""Exporter and validator for the VeRL SFT messages parquet baseline.

Production labels come EXCLUSIVELY from the canonical dataset contract:
records are read from the canonical all.json, only resolution_status ==
"resolved" records with a target whose category_id belongs to the
LeafRegistry enter training, and the ground truth is always
record["target"]["category_id"]. classification.level_1..level_4 stay
provenance only — there is deliberately NO fallback to them as labels.

Split boundaries come from one of two sources:
- canonical schema v2 embedded ``split`` fields (default; produced by
  script.canonical.split, which also carries per-record exclusion reasons);
- legacy split JSON files joined by record id (``split_dir`` argument).

Export gate (comparability guard): a label appearing in val or test but
absent from train fails the export unless explicitly waived via
``allow_label_gaps`` / ``allow_any_label_gap``. The waiver is recorded in
the report so no silent metric distortion survives.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from agent.hashing import sha256_file
from agent.task.contracts import GradingConfig

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

# Provenance marker for embedded-split sources (not a real filesystem path:
# a synthetic source label so error messages point at the canonical file).
_EMBEDDED_SPLIT_SOURCE_PREFIX = "#embedded-split="


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
    grading: GradingConfig | None = None,
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
    gt_level: str | None = None
    if grading is not None:
        gt_level = str(item.get(grading.gt_field, "") or "").strip()
        if not gt_level:
            raise ValueError(
                f"item {index} in {source} has no grading label under "
                f"{grading.gt_field!r}; such records must be excluded upstream"
            )
        if gt_level not in grading.levels:
            raise ValueError(
                f"item {index} in {source} has grading label {gt_level!r} outside "
                f"the configured levels {list(grading.levels)}"
            )
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
            grading=grading,
        )
        answer = stage2_answer(ground_truth, candidates, level=gt_level)
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
        "ground_truth_level": gt_level,
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
    split_dir: str | Path | None,
    output_dir: str | Path,
    registry: LeafRegistry | str | Path,
    task_config: TaskConfig | str | Path,
    corpus: Mapping[str, CorpusCategory],
    *,
    grading: GradingConfig | None = None,
    allow_label_gaps: Sequence[str] = (),
    allow_any_label_gap: bool = False,
) -> dict[str, Any]:
    """Export canonical records to one stage1 + one stage2 SFT row per split.

    Labels come exclusively from target.category_id (canonical contract).
    Split boundaries come from embedded schema-v2 ``split`` fields when
    ``split_dir`` is None, otherwise from the legacy split JSON files by
    record id. corpus is REQUIRED for the production Stage 2 path:
    candidates are resolved by category_id against the canonical corpus and
    there is no registry fallback in the exporter.

    Label-gap gate: a ground-truth label present in val/test but missing
    from train raises unless the label is whitelisted in
    ``allow_label_gaps`` or ``allow_any_label_gap`` is set; the report
    records the gate outcome either way.
    """
    leaf_registry = _registry(registry)
    config = _config(task_config)
    canonical_path = Path(canonical_file)
    output_root = Path(output_dir)
    split_root = None if split_dir is None else Path(split_dir)
    if split_dir is not None and not split_root.is_dir():
        raise FileNotFoundError(f"split directory not found: {split_root}")
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
        if split_dir is None:
            # schema v2 embedded splits must cover every resolved record;
            # otherwise the splitter was skipped or misconfigured
            assigned_split = str(item.get("split", "") or "").strip()
            if status == "resolved":
                if not assigned_split:
                    raise ValueError(
                        f"canonical record {item_id!r} is resolved without a "
                        "split assignment; run script.canonical.split first"
                    )
                if assigned_split not in SPLITS:
                    raise ValueError(
                        f"canonical record {item_id!r} carries unknown split "
                        f"{assigned_split!r}"
                    )
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
        "split_source": "embedded_v2" if split_dir is None else "split_dir_join",
        "metadata_fields": list(config.metadata_fields),
        "task_name": config.task_name,
        "registry_size": len(leaf_registry.categories),
        "corpus_size": len(corpus_map),
        "canonical_resolved": canonical_resolved,
        "idless_non_resolved_records": idless_non_resolved,
        "splits": {},
    }
    trainable_ids: set[str] = set()
    split_ground_truths: dict[str, set[str]] = {name: set() for name in SPLITS}
    split_levels: dict[str, set[str]] = {name: set() for name in SPLITS}
    all_rows: dict[str, list[dict[str, Any]]] = {}
    all_view_sizes: dict[str, int] = {}
    skipped_no_grading_label_counts: dict[str, int] = dict.fromkeys(SPLITS, 0)
    for split in SPLITS:
        if split_dir is None:
            # schema v2 embedded splits (script.canonical.split write-back)
            view: list[dict[str, Any]] = []
            for item in canonical_records:
                if str(item.get("split", "") or "").strip() != split:
                    continue
                status = str(item.get("resolution_status", "") or "").strip()
                if status != "resolved":
                    raise ValueError(
                        f"canonical record with non-resolved status {status!r} "
                        f"carries a {split!r} assignment; re-run script.canonical.split"
                    )
                view.append(item)
            source = canonical_path / f"{_EMBEDDED_SPLIT_SOURCE_PREFIX}{split}"
        else:
            source = split_root / f"{split}.json"
            view = load_json_records(source)
        all_view_sizes[split] = len(view)
        rows: list[dict[str, Any]] = []
        skipped_unresolved = 0
        skipped_no_grading_label = 0
        for index, item in enumerate(view):
            item_id = str(item.get("id", "") or "").strip()
            if not item_id:
                raise ValueError(f"split item {index} in {source} has no id")
            if grading is not None and not str(
                item.get(grading.gt_field, "") or ""
            ).strip():
                # joint head requires a level label; a resolved record without
                # one cannot produce a consistent stage1+stage2 pair
                skipped_no_grading_label += 1
                continue
            if split_dir is not None:
                # legacy join: fail fast when the split copy diverges from the
                # canonical dataset (unknown id, stale split dir)
                canonical_item = canonical_by_id.get(item_id)
                if canonical_item is None:
                    raise ValueError(
                        f"split item id {item_id!r} ({source}) is absent from the "
                        f"canonical dataset {canonical_path}"
                    )
                item = canonical_item
            ground_truth = _canonical_target(item, index, source, leaf_registry)
            if ground_truth is None:
                skipped_unresolved += 1
                continue
            trainable_ids.add(item_id)
            split_ground_truths[split].add(ground_truth)
            if grading is not None:
                split_levels[split].add(
                    str(item.get(grading.gt_field, "") or "").strip()
                )
            rows.extend(
                _row(
                    item,
                    source,
                    index,
                    leaf_registry,
                    config,
                    stage,
                    corpus_map,
                    grading=grading,
                )
                for stage in ("stage1", "stage2")
            )
        if not rows:
            raise ValueError(f"no resolved records available in {source}")
        all_rows[split] = rows
        skipped_no_grading_label_counts[split] = skipped_no_grading_label

    # label-gap export gate: val/test labels missing from train distort the
    # reference baseline; fail fast unless explicitly waived.
    waived_gaps: list[dict[str, str]] = []
    blocking_gaps: list[dict[str, str]] = []
    for later_split in ("val", "test"):
        for label in sorted(split_ground_truths[later_split] - split_ground_truths["train"]):
            entry = {"label": label, "split": later_split}
            if allow_any_label_gap or label in allow_label_gaps:
                waived_gaps.append(entry)
            else:
                blocking_gaps.append(entry)
    blocking_levels: list[dict[str, str]] = []
    waived_levels: list[dict[str, str]] = []
    if grading is not None:
        for later_split in ("val", "test"):
            for code in sorted(split_levels[later_split] - split_levels["train"]):
                entry = {"label": code, "split": later_split}
                if allow_any_label_gap or code in allow_label_gaps:
                    waived_levels.append(entry)
                else:
                    blocking_levels.append(entry)
    gate_status = "failed" if (blocking_gaps or blocking_levels) else (
        "waived" if (waived_gaps or waived_levels) else "passed"
    )
    report["label_gap_gate"] = {
        "status": gate_status,
        "blocking": blocking_gaps,
        "waived": waived_gaps,
        "blocking_levels": blocking_levels,
        "waived_levels": waived_levels,
    }
    if blocking_gaps or blocking_levels:
        raise ValueError(
            "label-gap gate failed; labels absent from train but present in "
            f"val/test: {blocking_gaps + blocking_levels}. Waive via "
            "allow_label_gaps/allow_any_label_gap after review."
        )
    report["grading"] = (
        {
            "enabled": True,
            "levels": list(grading.levels),
            "gt_field": grading.gt_field,
        }
        if grading is not None
        else {"enabled": False}
    )

    output_root.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        destination = output_root / f"{split}.parquet"
        _write_parquet(all_rows[split], destination)
        report["splits"][split] = {
            "split_records": all_view_sizes[split],
            "exported_records": len(all_rows[split]),
            "skipped_not_resolved": all_view_sizes[split]
            - len({row["source_id"] for row in all_rows[split]})
            - skipped_no_grading_label_counts[split],
            "skipped_no_grading_label": (
                skipped_no_grading_label_counts[split] if grading is not None else 0
            ),
            "output_file": str(destination),
            "parquet_sha256": sha256_file(destination),
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
    grading: GradingConfig | None = None,
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
        gt_level = row.get("ground_truth_level")
        level_mismatch = [
            e
            for e in (
                ["stage2 grading enabled but ground_truth_level missing"]
                if grading is not None
                and (not isinstance(gt_level, str) or not gt_level.strip())
                else []
            )
        ]
        errors.extend(level_mismatch)
        usable_level = gt_level if isinstance(gt_level, str) and gt_level.strip() else None
        evaluation = evaluate_stage2_choices(
            assistant,
            ground_truth=ground_truth,
            candidates=tuple(candidates),
            registry=registry,
            grading=grading,
            expected_level=usable_level if grading is not None else None,
        )
        errors.extend(f"stage2 evaluation: {error}" for error in evaluation.errors)
        if evaluation.contract_valid and not evaluation.correct:
            if grading is not None and not evaluation.level_correct:
                errors.append("stage2 level must equal ground_truth_level")
            else:
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
                grading=grading,
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
    grading: GradingConfig | None = None,
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
                    for error in _validate_row(
                        row, leaf_registry, config, corpus_map, grading=grading
                    ):
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
