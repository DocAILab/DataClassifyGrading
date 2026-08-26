"""Validate the formal shougang-only Stage1-only cascade release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

from agent.hashing import sha256_file
from agent.release_policy import (
    FORMAL_DATASETS,
    FORMAL_DATASET_SET,
    FORMAL_RELEASE_FORMAT,
    FORMAL_RELEASE_NAME,
    FORMAL_SAMPLING_POLICY,
)
from agent.task import LeafRegistry, TaskConfig
from agent.task.assets import load_corpus_categories
from agent.task.grading_manifest import DatasetGradingManifest
from agent.training.rl import validate_rl_row
from agent.training.rl.sample import NATIVE_TOOL_TRAJECTORY_FORMAT

_DATASETS = FORMAL_DATASETS
_SPLITS = ("train", "val", "test")


def expected_passthrough_materialization(
    input_counts: Mapping[str, int],
) -> tuple[dict[str, int], dict[str, float]]:
    """Return the singleton shougang counts and unit sampling weight."""

    if not isinstance(input_counts, Mapping) or set(input_counts) != FORMAL_DATASET_SET:
        raise ValueError(
            f"passthrough materialization requires exactly {FORMAL_RELEASE_NAME} count"
        )
    count = input_counts.get(FORMAL_RELEASE_NAME)
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("passthrough materialization count must be a positive integer")
    return {FORMAL_RELEASE_NAME: count}, {FORMAL_RELEASE_NAME: 1.0}


# Compatibility alias for callers that imported the old helper.  It is now a
# strict singleton check and intentionally rejects non-formal/joint inputs.
def expected_sqrt_materialization(
    input_counts: Mapping[str, int],
) -> tuple[dict[str, int], dict[str, float]]:
    return expected_passthrough_materialization(input_counts)


def _singleton_map(value: Any) -> bool:
    return isinstance(value, Mapping) and set(value) == FORMAL_DATASET_SET


def _valid_count(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _valid_weight(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and float(value) == 1.0
    )


def _check_count_maps(
    details: Mapping[str, Any],
    *,
    context: str,
    errors: list[str],
    actual: int | None = None,
) -> tuple[int | None, int | None]:
    """Validate singleton count/weight maps and return input/output counts."""

    input_counts = details.get("input_source_counts")
    source_counts = details.get("source_counts")
    achieved_weights = details.get("achieved_weights")
    if not _singleton_map(input_counts):
        errors.append(f"{context} input_source_counts must be exactly {{{FORMAL_RELEASE_NAME!r}}}")
    if not _singleton_map(source_counts):
        errors.append(f"{context} source_counts must be exactly {{{FORMAL_RELEASE_NAME!r}}}")
    if not _singleton_map(achieved_weights):
        errors.append(f"{context} achieved_weights must be exactly {{{FORMAL_RELEASE_NAME!r}}}")

    input_count: int | None = None
    source_count: int | None = None
    if _singleton_map(input_counts):
        raw = input_counts.get(FORMAL_RELEASE_NAME)
        if not _valid_count(raw):
            errors.append(f"{context} input count for {FORMAL_RELEASE_NAME} is invalid")
        else:
            input_count = raw
    if _singleton_map(source_counts):
        raw = source_counts.get(FORMAL_RELEASE_NAME)
        if not _valid_count(raw):
            errors.append(f"{context} source count for {FORMAL_RELEASE_NAME} is invalid")
        else:
            source_count = raw
    if _singleton_map(achieved_weights):
        if not _valid_weight(achieved_weights.get(FORMAL_RELEASE_NAME)):
            errors.append(f"{context} achieved weight for {FORMAL_RELEASE_NAME} must be 1.0")

    if input_count is not None and source_count is not None and input_count != source_count:
        errors.append(f"{context} source count must equal input count under passthrough policy")
    if actual is not None and source_count is not None and source_count != actual:
        errors.append(f"{context} source count mismatch for {FORMAL_RELEASE_NAME}")
    return input_count, source_count


def validate_cascade_release(
    dataset_dir: str | Path,
    *,
    registry: LeafRegistry,
    task_config: TaskConfig,
    corpus: Mapping[str, Any],
    grading_manifest: DatasetGradingManifest,
) -> dict[str, Any]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("cascade validation requires pyarrow") from exc

    root = Path(dataset_dir)
    report_path = root / "export_report.json"
    source_report = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(source_report, Mapping):
        raise ValueError("release export report must be an object")
    errors: list[str] = []

    missing_descriptions = sorted(
        category.category_id
        for category in registry.categories
        if not isinstance(category.description, str)
        or not category.description.strip()
    )
    if missing_descriptions:
        errors.append(
            "registry entries require non-empty descriptions: "
            + ", ".join(missing_descriptions)
        )

    try:
        grading = grading_manifest.config_for(FORMAL_RELEASE_NAME)
    except KeyError:
        grading = None
        errors.append(f"grading manifest has no {FORMAL_RELEASE_NAME} standard")
    if grading is not None and grading.gt_field != "data_level":
        errors.append(
            f"grading gt_field must be data_level for: {FORMAL_RELEASE_NAME}"
        )
    if task_config.metadata_fields != ("field_name", "table_name"):
        errors.append(
            "task config metadata_fields must be field_name+table_name "
            "(contract change 2025-08-25)"
        )

    if source_report.get("family") != "rl-cascade":
        errors.append("release family must be rl-cascade")
    if source_report.get("trajectory_format") != NATIVE_TOOL_TRAJECTORY_FORMAT:
        errors.append(
            f"release trajectory_format must be {NATIVE_TOOL_TRAJECTORY_FORMAT}"
        )
    if source_report.get("dataset") != FORMAL_RELEASE_NAME:
        errors.append(f"release dataset must be exactly {FORMAL_RELEASE_NAME}")
    if source_report.get("format") != FORMAL_RELEASE_FORMAT:
        errors.append(
            f"release report format must be {FORMAL_RELEASE_FORMAT}"
        )
    release = source_report.get("release")
    if (
        not isinstance(release, Mapping)
        or release.get("status") != "passed"
        or release.get("published") is not True
    ):
        errors.append("release must be passed and published")

    sampling_train_source_count: int | None = None
    sampling = source_report.get("sampling")
    expected_sampling_fields = {
        "policy",
        "train_input_source_counts",
        "train_source_counts",
        "train_achieved_weights",
    }
    if not isinstance(sampling, Mapping) or set(sampling) != expected_sampling_fields:
        errors.append("release sampling object is malformed")
    elif sampling.get("policy") != FORMAL_SAMPLING_POLICY:
        errors.append(
            f"release must record the approved {FORMAL_SAMPLING_POLICY} policy"
        )
    else:
        train_input, train_source = _check_count_maps(
            {
                "input_source_counts": sampling.get("train_input_source_counts"),
                "source_counts": sampling.get("train_source_counts"),
                "achieved_weights": sampling.get("train_achieved_weights"),
            },
            context="sampling",
            errors=errors,
        )
        if train_input is not None and train_source is not None:
            sampling_train_source_count = train_source
            if train_input != train_source:
                errors.append("sampling source count must equal input count")

    gate = source_report.get("label_gap_gate")
    if not isinstance(gate, Mapping) or gate.get("status") != "passed":
        errors.append("cascade release label/level gap gate must be passed")
    elif any(
        gate.get(name)
        for name in ("blocking", "blocking_levels", "waived", "waived_levels")
    ):
        errors.append("cascade release label/level gap gate must contain no gaps or waivers")

    raw_manifest = source_report.get("grading_manifest")
    if not isinstance(raw_manifest, Mapping):
        errors.append("cascade release must carry embedded grading_manifest lineage")
    else:
        manifest_path = raw_manifest.get("path")
        manifest_sha = raw_manifest.get("sha256")
        if not isinstance(manifest_path, str) or not isinstance(manifest_sha, str):
            errors.append("grading manifest lineage is incomplete")
        elif not Path(manifest_path).is_file() or sha256_file(manifest_path) != manifest_sha:
            errors.append("grading manifest sha256 mismatch")
        manifest_datasets = raw_manifest.get("datasets")
        if not _singleton_map(manifest_datasets):
            errors.append(
                "grading manifest lineage must contain exactly the shougang dataset"
            )
        elif grading is not None:
            details = manifest_datasets.get(FORMAL_RELEASE_NAME)
            expected = details.get("sha256") if isinstance(details, Mapping) else None
            try:
                expected_hash = grading_manifest.sha256_for(FORMAL_RELEASE_NAME)
            except KeyError:
                expected_hash = None
            if expected != expected_hash:
                errors.append(
                    f"{FORMAL_RELEASE_NAME} grading standard hash mismatch"
                )

    raw_inputs = source_report.get("inputs")
    if not _singleton_map(raw_inputs):
        errors.append(
            "cascade release must carry exactly the shougang input release lineage"
        )
        raw_inputs = {}
    else:
        lineage = raw_inputs.get(FORMAL_RELEASE_NAME)
        if not isinstance(lineage, Mapping):
            errors.append(f"{FORMAL_RELEASE_NAME} input release lineage is malformed")
        else:
            report_path_raw = lineage.get("export_report_path")
            report_sha = lineage.get("export_report_sha256")
            parquet_paths = lineage.get("parquet_paths")
            parquet_hashes = lineage.get("parquet_sha256")
            if not isinstance(report_path_raw, str) or not isinstance(report_sha, str):
                errors.append(
                    f"{FORMAL_RELEASE_NAME} input release report hash lineage is missing"
                )
            else:
                report_file = Path(report_path_raw)
                if not report_file.is_file() or sha256_file(report_file) != report_sha:
                    errors.append(
                        f"{FORMAL_RELEASE_NAME} input release report hash mismatch"
                    )
            if (
                not isinstance(parquet_paths, Mapping)
                or set(parquet_paths) != set(_SPLITS)
                or not isinstance(parquet_hashes, Mapping)
                or set(parquet_hashes) != set(_SPLITS)
            ):
                errors.append(
                    f"{FORMAL_RELEASE_NAME} input parquet hash lineage is missing or not split-complete"
                )
            else:
                for split in _SPLITS:
                    parquet_raw = parquet_paths.get(split)
                    expected = parquet_hashes.get(split)
                    if not isinstance(parquet_raw, str) or not isinstance(expected, str):
                        errors.append(
                            f"{FORMAL_RELEASE_NAME} input {split} hash lineage is malformed"
                        )
                        continue
                    parquet_file = Path(parquet_raw)
                    if not parquet_file.is_file() or sha256_file(parquet_file) != expected:
                        errors.append(
                            f"{FORMAL_RELEASE_NAME} input {split} parquet hash mismatch"
                        )

    raw_splits = source_report.get("splits")
    if not isinstance(raw_splits, Mapping) or set(raw_splits) != set(_SPLITS):
        errors.append("release report must contain exactly train, val, and test splits")
        raw_splits = {}

    all_ids: dict[str, set[str]] = {}
    datasets_seen: set[str] = set()
    stage2_rows = 0
    duplicate_source_ids = 0
    rows_by_split: dict[str, int] = {}
    rows_by_dataset: dict[str, dict[str, int]] = {
        split: {FORMAL_RELEASE_NAME: 0} for split in _SPLITS
    }
    for split in _SPLITS:
        parquet = root / f"{split}.parquet"
        details = raw_splits.get(split)
        if not parquet.is_file() or not isinstance(details, Mapping):
            errors.append(f"{split} artifact/report is missing")
            all_ids[split] = set()
            continue
        if details.get("parquet_sha256") != sha256_file(parquet):
            errors.append(f"{split} parquet sha256 mismatch")
        rows = pq.read_table(parquet).to_pylist()
        rows_by_split[split] = len(rows)
        ids: list[str] = []
        split_datasets: set[str] = set()
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                errors.append(f"{split} row {index} is not an object")
                continue
            extra = row.get("extra_info")
            dataset = extra.get("dataset") if isinstance(extra, Mapping) else None
            stage = extra.get("stage") if isinstance(extra, Mapping) else None
            source_id = extra.get("source_id") if isinstance(extra, Mapping) else None
            if dataset != FORMAL_RELEASE_NAME:
                errors.append(
                    f"{split} row {index} has invalid dataset; only {FORMAL_RELEASE_NAME} is allowed"
                )
                continue
            split_datasets.add(dataset)
            datasets_seen.add(dataset)
            rows_by_dataset[split][FORMAL_RELEASE_NAME] += 1
            if stage != "stage1":
                stage2_rows += 1
                errors.append(f"{split} row {index} is not Stage1-only")
            if isinstance(source_id, str) and source_id.strip():
                ids.append(source_id.strip())
            if grading is not None:
                row_errors = validate_rl_row(
                    row,
                    dataset=FORMAL_RELEASE_NAME,
                    registry=registry,
                    task_config=task_config,
                    corpus=corpus,
                    grading=grading,
                )
                if row_errors:
                    errors.append(
                        f"{split} row {index} violates {len(row_errors)} RL contract checks"
                    )
        duplicate_source_ids += len(ids) - len(set(ids))
        if len(ids) != len(set(ids)):
            errors.append(f"{split} contains duplicate source ids")
        if split_datasets != FORMAL_DATASET_SET:
            errors.append(
                f"{split} rows must contain only the {FORMAL_RELEASE_NAME} dataset"
            )
        all_ids[split] = set(ids)

        # The split report is part of the passthrough contract, not an
        # unchecked annotation.  Every map must be a singleton and the output
        # count must equal both the input count and actual Stage1 rows.
        _check_count_maps(
            details,
            context=f"{split} report",
            errors=errors,
            actual=rows_by_dataset[split][FORMAL_RELEASE_NAME],
        )

    actual_train_count = rows_by_dataset["train"][FORMAL_RELEASE_NAME]
    if (
        sampling_train_source_count is not None
        and sampling_train_source_count != actual_train_count
    ):
        errors.append("sampling train source count does not match actual train rows")

    cross_split_overlap = 0
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = all_ids.get(left, set()) & all_ids.get(right, set())
        cross_split_overlap += len(overlap)
    if cross_split_overlap:
        errors.append("source ids overlap across splits")

    return {
        "format": "dataclassify-cascade-release-validation-v1",
        "valid": not errors,
        "datasets": sorted(datasets_seen),
        "rows": rows_by_split,
        "rows_by_dataset": rows_by_dataset,
        "stage2_rows": stage2_rows,
        "duplicate_source_ids": duplicate_source_ids,
        "cross_split_source_id_overlap": cross_split_overlap,
        "error_count": len(errors),
        "errors": errors,
    }


def _args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--task-config", required=True)
    parser.add_argument("--grading-manifest", required=True)
    parser.add_argument("--report")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _args(argv)
    try:
        registry = LeafRegistry.from_path(args.registry)
        task = TaskConfig.from_path(args.task_config)
        grading_manifest = DatasetGradingManifest.from_path(args.grading_manifest)
        corpus = {
            item.category_id: item
            for item in load_corpus_categories(args.corpus)
        }
        report = validate_cascade_release(
            args.dataset_dir,
            registry=registry,
            task_config=task,
            corpus=corpus,
            grading_manifest=grading_manifest,
        )
        if args.report:
            output = Path(args.report)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"validate_cascade: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
