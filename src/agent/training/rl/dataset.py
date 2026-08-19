"""Exporter and validator for the VeRL RL parquet (v0.8.0 five-field schema).

RL rows are built from the same canonical contract as SFT: only
resolution_status == "resolved" records with target.category_id in the
LeafRegistry enter, split boundaries follow the original split JSON files
by record id, and classification level_1..level_4 stay provenance.

VeRL v0.8.0 RL parquet columns (verified against verl v0.8.0 RLHFDataset and
the reward manager, which reads ``non_tensor_batch["reward_model"]
["ground_truth"]``):

- data_source: ``<dataset>/stage1`` or ``<dataset>/stage2`` — routes the
  reward function.
- prompt: list of chat messages [system, user]; there is NO assistant gold
  response (rollout appends the generation prompt).
- ability: task category (task_config.task_name).
- reward_model: {"style": "rule", "ground_truth": "<category_id>"}.
- extra_info: dataset / stage / source_id / prompt-visible metadata, plus
  the Stage 2 candidate bundle (fixture policy, not production retrieval).

No training-algorithm internal fields are stored.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from agent.task.contracts import CorpusCategory, LeafRegistry, TaskConfig
from agent.task.prompts import build_stage1_prompt, build_stage2_prompt
from agent.training.common import (
    canonical_target,
    require_corpus,
    require_corpus_covers_registry,
)
from agent.training.rl.sample import build_rl_row, build_rl_samples

RL_SPLITS = ("train", "val", "test")
VERL_RL_COLUMNS = ("data_source", "prompt", "ability", "reward_model", "extra_info")


def _registry(value: LeafRegistry | str | Path) -> LeafRegistry:
    return value if isinstance(value, LeafRegistry) else LeafRegistry.from_path(value)


def _config(value: TaskConfig | str | Path) -> TaskConfig:
    return value if isinstance(value, TaskConfig) else TaskConfig.from_path(value)


def _write_parquet(rows: list[dict[str, Any]], path: Path) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - dependency install issue
        raise RuntimeError("RL export requires pyarrow") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


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


def export_rl_dataset(
    canonical_file: str | Path,
    split_dir: str | Path,
    output_dir: str | Path,
    dataset: str,
    registry: LeafRegistry | str | Path,
    task_config: TaskConfig | str | Path,
    corpus: Mapping[str, CorpusCategory],
) -> dict[str, Any]:
    """Export canonical records to VeRL RL parquet (one file per split).

    Labels come exclusively from target.category_id (canonical contract);
    split boundaries follow the original split JSON files by record id.
    corpus is REQUIRED: Stage 2 candidates are resolved by category_id
    against the canonical corpus (no registry fallback).
    """
    if not isinstance(dataset, str) or not dataset.strip():
        raise ValueError("dataset name must be a non-empty string")
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
            canonical_target(item, index, canonical_path, leaf_registry)
            if not item_id:
                raise ValueError(
                    f"resolved canonical record without id: {canonical_path}"
                )
        elif not item_id:
            # stage-3B contract allows id-less audit records for non-resolved
            # outcomes; they never enter training and are ignored by the
            # split join, but must not fail the export
            idless_non_resolved += 1
            continue
        if item_id in canonical_by_id:
            raise ValueError(f"duplicate id in canonical dataset: {item_id}")
        canonical_by_id[item_id] = item
    corpus_map = require_corpus(corpus)
    require_corpus_covers_registry(corpus_map, leaf_registry)

    report: dict[str, Any] = {
        "format": "verl_rl_parquet",
        "version": "verl 0.8.0 five-field schema (data_source/prompt/ability/reward_model/extra_info)",
        "label_source": "canonical target.category_id (resolution_status == resolved)",
        "candidate_policy": (
            "baseline fixture: (ground_truth + first four registry negatives) "
            "permuted deterministically by stable source_id (not position-fixed, "
            "not the production retrieval policy)"
        ),
        "dataset": dataset,
        "metadata_fields": list(config.metadata_fields),
        "task_name": config.task_name,
        "registry_size": len(leaf_registry.categories),
        "corpus_size": len(corpus_map),
        "canonical_resolved": canonical_resolved,
        "idless_non_resolved_records": idless_non_resolved,
        "splits": {},
    }
    trainable_ids: set[str] = set()
    for split in RL_SPLITS:
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
            status = str(canonical_item.get("resolution_status", "") or "").strip()
            if status != "resolved":
                skipped_unresolved += 1
                continue
            trainable_ids.add(item_id)
            stage1, stage2 = build_rl_samples(
                canonical_item,
                index,
                source,
                dataset=dataset,
                registry=leaf_registry,
                config=config,
                corpus=corpus_map,
            )
            rows.extend(build_rl_row(sample, config) for sample in (stage1, stage2))
        if not rows:
            raise ValueError(f"no resolved records available in {source}")
        destination = output_root / f"{split}.parquet"
        _write_parquet(rows, destination)
        report["splits"][split] = {
            "split_records": len(split_records),
            "exported_rows": len(rows),
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
    *,
    dataset: str,
    registry: LeafRegistry,
    config: TaskConfig,
    corpus: Mapping[str, CorpusCategory],
) -> list[str]:
    """Validate one RL parquet row against the VeRL v0.8.0 contract."""
    errors: list[str] = []

    if set(row) != set(VERL_RL_COLUMNS):
        errors.append(
            f"row must contain exactly the columns {sorted(VERL_RL_COLUMNS)}, "
            f"got {sorted(row)}"
        )
        return errors

    data_source = row.get("data_source")
    if not isinstance(data_source, str) or not data_source.strip():
        errors.append("data_source must be a non-empty string")
    elif data_source not in {f"{dataset}/stage1", f"{dataset}/stage2"}:
        errors.append(f"data_source must be {dataset}/stage1 or {dataset}/stage2")

    prompt = row.get("prompt")
    roles: list[str] = []
    contents: list[str] = []
    if not isinstance(prompt, list):
        errors.append("prompt must be a list of chat messages")
    else:
        for message in prompt:
            if not isinstance(message, Mapping) or not all(
                key in message for key in ("role", "content")
            ):
                errors.append("prompt messages must have role and content")
                continue
            roles.append(str(message.get("role", "")))
            contents.append(str(message.get("content", "")))
        if roles != ["system", "user"]:
            errors.append("prompt roles must be exactly [system, user] (no assistant gold)")
        if any(not content for content in contents):
            errors.append("prompt message content must be non-empty")
        if any(
            token in "".join(contents)
            for token in ("<|im_start|>", "<|im_end|>")
        ):
            errors.append("prompt must not contain serialized chat-template tokens")

    ability = row.get("ability")
    if not isinstance(ability, str) or ability != config.task_name:
        errors.append(f"ability must equal task_name {config.task_name!r}")

    reward_model = row.get("reward_model")
    ground_truth: str | None = None
    if not isinstance(reward_model, Mapping):
        errors.append("reward_model must be a dict")
    else:
        style = reward_model.get("style")
        if style != "rule":
            errors.append("reward_model.style must be 'rule'")
        raw_gt = reward_model.get("ground_truth")
        if not isinstance(raw_gt, str) or not raw_gt.strip():
            errors.append("reward_model.ground_truth must be a non-empty string")
        elif raw_gt not in registry.ids:
            errors.append(f"reward_model.ground_truth {raw_gt!r} is absent from the leaf registry")
        else:
            ground_truth = raw_gt
        unknown_keys = set(reward_model) - {"style", "ground_truth"}
        if unknown_keys:
            errors.append(f"reward_model contains unexpected keys: {sorted(unknown_keys)}")

    extra_info = row.get("extra_info")
    stage: str | None = None
    source_id: str | None = None
    candidates: list[str] | None = None
    metadata: dict[str, Any] | None = None
    if not isinstance(extra_info, Mapping):
        errors.append("extra_info must be a dict")
    else:
        if extra_info.get("dataset") != dataset:
            errors.append(f"extra_info.dataset must equal {dataset!r}")
        raw_stage = extra_info.get("stage")
        if raw_stage not in {"stage1", "stage2"}:
            errors.append("extra_info.stage must be stage1 or stage2")
        else:
            stage = raw_stage
        raw_source_id = extra_info.get("source_id")
        if not isinstance(raw_source_id, str) or not raw_source_id.strip():
            errors.append("extra_info.source_id must be a non-empty string")
        else:
            source_id = raw_source_id
        raw_metadata = extra_info.get("metadata")
        if not isinstance(raw_metadata, Mapping) or set(raw_metadata) != set(config.metadata_fields):
            errors.append("extra_info.metadata must exactly match task config metadata_fields")
        else:
            metadata = dict(raw_metadata)
        raw_candidates = extra_info.get("candidates")
        if raw_stage == "stage1":
            if raw_candidates is not None:
                errors.append("stage1 extra_info must not carry a candidate bundle")
        elif raw_stage == "stage2":
            if (
                not isinstance(raw_candidates, list)
                or len(raw_candidates) != 5
                or not all(isinstance(candidate, str) for candidate in raw_candidates)
                or len(set(raw_candidates)) != 5
            ):
                errors.append("stage2 extra_info.candidates must be exactly 5 unique strings")
            elif any(candidate not in registry.ids for candidate in raw_candidates):
                errors.append("stage2 extra_info.candidates contain an ID absent from registry")
            else:
                candidates = raw_candidates
        unknown_keys = set(extra_info) - {"dataset", "stage", "source_id", "metadata", "candidates"}
        if unknown_keys:
            errors.append(f"extra_info contains unexpected keys: {sorted(unknown_keys)}")
    if data_source and stage and data_source != f"{dataset}/{stage}":
        errors.append("data_source stage suffix must match extra_info.stage")

    if ground_truth is not None and metadata is not None and roles == ["system", "user"]:
        expected_prompt = None
        if stage == "stage1":
            expected_prompt = build_stage1_prompt(metadata, registry, config)
        elif stage == "stage2" and candidates is not None:
            expected_prompt = build_stage2_prompt(
                metadata, candidates, registry, config, corpus=corpus or None
            )
        if expected_prompt is not None and contents[:2] != [
            expected_prompt.system,
            expected_prompt.user,
        ]:
            errors.append("prompt does not match registry and task contract")
        if stage == "stage1" and contents:
            user = contents[1]
            if '"category_id"' in user:
                errors.append("stage1 prompt must not expose canonical category ids")
            try:
                catalog = json.loads(
                    user.split("\n", 1)[1].split("\nField metadata:", 1)[0]
                )
            except (json.JSONDecodeError, AttributeError):
                errors.append("stage1 prompt catalog must be a JSON array")
            else:
                if (
                    not isinstance(catalog, list)
                    or len(catalog) != len(registry.categories)
                    or any(
                        not (isinstance(entry, list) and len(entry) == 2)
                        for entry in catalog
                    )
                ):
                    errors.append("stage1 prompt must render the full leaf registry catalog")
    return errors


def _validate_stage_pairs(rows: list[Mapping[str, Any]]) -> list[str]:
    """Every source_id must have exactly one stage1 row AND one stage2 row.

    Rows are grouped into lists per (source_id, stage) so duplicate stage
    rows are detected instead of silently overwriting each other:
    - stage1 + stage1 + stage2 -> failure (two stage1 rows)
    - stage1 + stage2 + stage2 -> failure (two stage2 rows)
    - only stage1 / only stage2  -> failure (missing one stage).
    """
    grouped: dict[str, dict[str, list[Mapping[str, Any]]]] = {}
    for row in rows:
        extra_info = row.get("extra_info")
        source_id = extra_info.get("source_id") if isinstance(extra_info, Mapping) else None
        stage = extra_info.get("stage") if isinstance(extra_info, Mapping) else None
        if isinstance(source_id, str) and source_id.strip() and stage in {"stage1", "stage2"}:
            grouped.setdefault(source_id, {}).setdefault(stage, []).append(row)
    errors: list[str] = []
    for source_id, by_stage in grouped.items():
        count_stage1 = len(by_stage.get("stage1", []))
        count_stage2 = len(by_stage.get("stage2", []))
        if count_stage1 != 1 or count_stage2 != 1:
            errors.append(
                f"source_id {source_id!r} must have exactly one stage1 row and one "
                f"stage2 row (found {count_stage1} stage1, {count_stage2} stage2)"
            )
            continue
        ground_truths = {
            row.get("reward_model", {}).get("ground_truth")
            if isinstance(row.get("reward_model"), Mapping)
            else None
            for row in by_stage["stage1"] + by_stage["stage2"]
        }
        if len(ground_truths) != 1:
            errors.append(f"source_id {source_id!r} must share one ground_truth across stages")
    return errors


def validate_rl_dataset(
    dataset_dir: str | Path,
    dataset: str,
    registry: LeafRegistry | str | Path,
    task_config: TaskConfig | str | Path,
    corpus: Mapping[str, CorpusCategory],
) -> dict[str, Any]:
    """Return a structured validation report; malformed rows are reported,
    not silently accepted."""
    leaf_registry = _registry(registry)
    config = _config(task_config)
    corpus_map = require_corpus(corpus)
    require_corpus_covers_registry(corpus_map, leaf_registry)
    root = Path(dataset_dir)
    report: dict[str, Any] = {
        "format": "verl_rl_parquet",
        "valid": True,
        "dataset": dataset,
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
        raise RuntimeError("RL validation requires pyarrow") from exc
    source_ids_by_split: dict[str, set[str]] = {}
    for split in RL_SPLITS:
        path = root / f"{split}.parquet"
        details: dict[str, Any] = {"rows": 0, "errors": []}
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
                        row,
                        dataset=dataset,
                        registry=leaf_registry,
                        config=config,
                        corpus=corpus_map,
                    ):
                        details["errors"].append(f"row {index}: {error}")
                details["errors"].extend(_validate_stage_pairs(rows))
            except Exception as exc:  # malformed parquet is a validation failure
                details["errors"].append(f"cannot read parquet: {exc}")
        source_ids_by_split[split] = {
            source_id
            for row in rows
            if isinstance(
                (extra_info := row.get("extra_info")), Mapping
            )
            and isinstance((source_id := extra_info.get("source_id")), str)
            and source_id.strip()
        }
        report["splits"][split] = details
        if details["errors"]:
            report["valid"] = False

    for left_index, left in enumerate(RL_SPLITS):
        for right in RL_SPLITS[left_index + 1 :]:
            overlap = source_ids_by_split[left] & source_ids_by_split[right]
            if overlap:
                examples = ", ".join(sorted(overlap)[:5])
                report["cross_split_errors"].append(
                    f"source_id overlap between {left} and {right}: {examples}"
                )
    if report["cross_split_errors"]:
        report["valid"] = False
    return report


__all__ = [
    "RL_SPLITS",
    "VERL_RL_COLUMNS",
    "export_rl_dataset",
    "validate_rl_dataset",
]
