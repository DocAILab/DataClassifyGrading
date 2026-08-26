"""Exporter and validator for the VeRL 0.9 native-tool RL parquet.

RL rows are built from the same canonical contract as SFT: only
resolution_status == "resolved" records with target.category_id in the
LeafRegistry enter, split boundaries follow the original split JSON files
by record id, and classification level_1..level_4 stay provenance.

VeRL v0.9.0 RL parquet columns (consumed by RLHFDataset and the native
ToolAgentLoop; the reward target remains in ``non_tensor_batch["reward_model"]
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
from typing import Any, Mapping, Sequence

from agent.hashing import sha256_file
from agent.task.contracts import CorpusCategory, GradingConfig, LeafRegistry, TaskConfig
from agent.task.prompts import build_stage1_prompt, build_stage2_prompt
from agent.training.common import (
    canonical_target,
    require_corpus,
    require_corpus_covers_registry,
)
from agent.training.rl.sample import (
    NATIVE_TOOL_TRAJECTORY_FORMAT,
    build_native_tool_prompt,
    build_rl_row,
    build_rl_samples,
)

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
    split_dir: str | Path | None,
    output_dir: str | Path,
    dataset: str,
    registry: LeafRegistry | str | Path,
    task_config: TaskConfig | str | Path,
    corpus: Mapping[str, CorpusCategory],
    *,
    grading: GradingConfig | None = None,
    allow_label_gaps: Sequence[str] = (),
    allow_any_label_gap: bool = False,
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
    canonical_path = Path(canonical_file)
    split_root = None if split_dir is None else Path(split_dir)
    output_root = Path(output_dir)
    if split_root is not None and not split_root.is_dir():
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
            canonical_target(item, index, canonical_path, leaf_registry)
            if not item_id:
                raise ValueError(
                    f"resolved canonical record without id: {canonical_path}"
                )
            if split_root is None:
                assigned_split = str(item.get("split", "") or "").strip()
                if not assigned_split:
                    raise ValueError(
                        f"canonical record {item_id!r} is resolved without a split assignment"
                    )
                if assigned_split not in RL_SPLITS:
                    raise ValueError(
                        f"canonical record {item_id!r} carries unknown split {assigned_split!r}"
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
        "version": "verl 0.9.0 native-tool five-field schema",
        "trajectory_format": NATIVE_TOOL_TRAJECTORY_FORMAT,
        "label_source": "canonical target.category_id (resolution_status == resolved)",
        "candidate_policy": (
            "baseline fixture: (ground_truth + first four registry negatives) "
            "permuted deterministically by stable source_id (not position-fixed, "
            "not the production retrieval policy)"
        ),
        "split_source": "embedded_v2" if split_root is None else "split_dir_join",
        "dataset": dataset,
        "metadata_fields": list(config.metadata_fields),
        "task_name": config.task_name,
        "registry_size": len(leaf_registry.categories),
        "corpus_size": len(corpus_map),
        "canonical_resolved": canonical_resolved,
        "idless_non_resolved_records": idless_non_resolved,
        "grading": (
            {
                "enabled": True,
                "levels": list(grading.levels),
                "gt_field": grading.gt_field,
            }
            if grading is not None
            else {"enabled": False}
        ),
        "splits": {},
    }
    trainable_ids: set[str] = set()
    split_ground_truths: dict[str, set[str]] = {split: set() for split in RL_SPLITS}
    split_levels: dict[str, set[str]] = {split: set() for split in RL_SPLITS}
    for split in RL_SPLITS:
        if split_root is None:
            split_records = [
                item for item in canonical_records
                if str(item.get("split", "") or "").strip() == split
            ]
            source = canonical_path
        else:
            source = split_root / f"{split}.json"
            split_records = load_json_records(source)
        rows: list[dict[str, Any]] = []
        skipped_unresolved = 0
        skipped_no_grading_label = 0
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
            if split_root is not None:
                canonical_item = canonical_item
            status = str(canonical_item.get("resolution_status", "") or "").strip()
            if status != "resolved":
                skipped_unresolved += 1
                continue
            if grading is not None:
                raw_level = canonical_item.get(grading.gt_field, "")
                level = "" if raw_level is None else str(raw_level).strip()
                if not level:
                    skipped_no_grading_label += 1
                    continue
                if level not in grading.levels:
                    raise ValueError(
                        f"record {item_id!r} has grading label {level!r} outside "
                        f"configured levels {list(grading.levels)}"
                    )
            ground_truth = canonical_target(canonical_item, index, source, leaf_registry)
            assert ground_truth is not None
            trainable_ids.add(item_id)
            split_ground_truths[split].add(ground_truth)
            if grading is not None:
                split_levels[split].add(level)
            stage1, stage2 = build_rl_samples(
                canonical_item,
                index,
                source,
                dataset=dataset,
                registry=leaf_registry,
                config=config,
                corpus=corpus_map,
                grading=grading,
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
            "skipped_no_grading_label": skipped_no_grading_label,
            "output_file": str(destination),
            "parquet_sha256": sha256_file(destination),
        }
    blocking_gaps: list[dict[str, str]] = []
    waived_gaps: list[dict[str, str]] = []
    blocking_levels: list[dict[str, str]] = []
    waived_levels: list[dict[str, str]] = []
    allowed_gaps = {str(label) for label in allow_label_gaps}
    for later_split in ("val", "test"):
        for label in sorted(split_ground_truths[later_split] - split_ground_truths["train"]):
            entry = {"label": label, "split": later_split}
            if allow_any_label_gap or label in allowed_gaps:
                waived_gaps.append(entry)
            else:
                blocking_gaps.append(entry)
        if grading is not None:
            for level in sorted(split_levels[later_split] - split_levels["train"]):
                entry = {"label": level, "split": later_split}
                if allow_any_label_gap or level in allowed_gaps:
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
            "label-gap gate failed; labels absent from train but present in val/test: "
            f"{blocking_gaps + blocking_levels}. Waive via allow_label_gaps/allow_any_label_gap."
        )
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
    grading: GradingConfig | None = None,
) -> list[str]:
    """Validate one RL parquet row against the VeRL v0.9.0 contract."""
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
    ground_truth_level: str | None = None
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
        raw_level = extra_info.get("ground_truth_level")
        if raw_level is not None:
            if not isinstance(raw_level, str) or not raw_level.strip():
                errors.append("extra_info.ground_truth_level must be non-empty when provided")
            else:
                ground_truth_level = raw_level.strip()
                if grading is not None and ground_truth_level not in grading.levels:
                    errors.append(
                        f"extra_info.ground_truth_level {ground_truth_level!r} is not in "
                        f"configured levels {list(grading.levels)}"
                    )
        if grading is not None and not ground_truth_level:
            errors.append("grading enabled but extra_info.ground_truth_level is missing")
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
        trajectory_format = extra_info.get("trajectory_format")
        if raw_stage == "stage1":
            if grading is not None and trajectory_format != NATIVE_TOOL_TRAJECTORY_FORMAT:
                errors.append(
                    f"stage1 trajectory_format must be {NATIVE_TOOL_TRAJECTORY_FORMAT}"
                )
            elif grading is None and trajectory_format is not None:
                errors.append("classification-only stage1 must not carry trajectory_format")
        elif trajectory_format is not None:
            errors.append("stage2 fixture row must not carry trajectory_format")
        unknown_keys = set(extra_info) - {
            "dataset", "stage", "source_id", "metadata", "candidates",
            "ground_truth_level", "trajectory_format",
        }
        if unknown_keys:
            errors.append(f"extra_info contains unexpected keys: {sorted(unknown_keys)}")
    if data_source and stage and data_source != f"{dataset}/{stage}":
        errors.append("data_source stage suffix must match extra_info.stage")

    if ground_truth is not None and metadata is not None and roles == ["system", "user"]:
        expected_prompt = None
        if stage == "stage1":
            expected_prompt = (
                build_native_tool_prompt(metadata, grading, registry)
                if grading is not None
                else build_stage1_prompt(metadata, registry, config)
            )
        elif stage == "stage2" and candidates is not None:
            expected_prompt = build_stage2_prompt(
                metadata,
                candidates,
                registry,
                config,
                corpus=corpus or None,
                grading=grading,
            )
        if expected_prompt is not None and contents[:2] != [
            expected_prompt.system,
            expected_prompt.user,
        ]:
            errors.append("prompt does not match registry and task contract")
        if stage == "stage1" and contents:
            joined_prompt = "\n".join(contents)
            if any(category_id in joined_prompt for category_id in registry.ids):
                errors.append("stage1 prompt must not expose canonical category ids")
            if grading is not None:
                if "catalog" in contents[1].casefold() or '"candidates"' in joined_prompt:
                    errors.append("native-tool prompt must not embed a category catalog or candidates")
                try:
                    visible = json.loads(contents[1].split("\n", 1)[1])
                except (json.JSONDecodeError, AttributeError, IndexError):
                    errors.append("native-tool user prompt must end with metadata JSON")
                else:
                    if visible != metadata:
                        errors.append("native-tool user prompt metadata does not match extra_info")
            else:
                try:
                    catalog = json.loads(
                        contents[1].split("\n", 1)[1].split("\nField metadata:", 1)[0]
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


def validate_rl_row(
    row: Mapping[str, Any],
    *,
    dataset: str,
    registry: LeafRegistry,
    task_config: TaskConfig,
    corpus: Mapping[str, CorpusCategory],
    grading: GradingConfig | None = None,
) -> list[str]:
    """Validate one five-field row through the public RL contract seam."""

    return _validate_row(
        row,
        dataset=dataset,
        registry=registry,
        config=task_config,
        corpus=corpus,
        grading=grading,
    )


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
        levels = {
            row.get("extra_info", {}).get("ground_truth_level")
            if isinstance(row.get("extra_info"), Mapping)
            else None
            for row in by_stage["stage1"] + by_stage["stage2"]
        }
        if len(levels) != 1:
            errors.append(f"source_id {source_id!r} must share one ground_truth_level across stages")
    return errors


def validate_rl_dataset(
    dataset_dir: str | Path,
    dataset: str,
    registry: LeafRegistry | str | Path,
    task_config: TaskConfig | str | Path,
    corpus: Mapping[str, CorpusCategory],
    *,
    grading: GradingConfig | None = None,
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
        "grading": (
            {
                "enabled": True,
                "levels": list(grading.levels),
                "gt_field": grading.gt_field,
            }
            if grading is not None
            else {"enabled": False}
        ),
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
                        grading=grading,
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
    "validate_rl_row",
    "validate_rl_dataset",
]
