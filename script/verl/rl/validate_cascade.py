"""Validate a formal finance+shougang Stage1-only cascade mixture release."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping

from agent.hashing import sha256_file
from agent.task import LeafRegistry, TaskConfig
from agent.task.assets import load_corpus_categories
from agent.task.grading_manifest import DatasetGradingManifest
from agent.training.rl import validate_rl_row

_DATASETS = ("finance", "shougang")
_SPLITS = ("train", "val", "test")


def expected_sqrt_materialization(
    input_counts: Mapping[str, int],
) -> tuple[dict[str, int], dict[str, float]]:
    """Return exact ceil materialization counts and achieved proportions."""

    if set(input_counts) != set(_DATASETS):
        raise ValueError("sqrt materialization requires finance and shougang counts")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in input_counts.values()
    ):
        raise ValueError("sqrt materialization counts must be positive integers")
    roots = {dataset: math.sqrt(input_counts[dataset]) for dataset in _DATASETS}
    root_total = sum(roots.values())
    raw_weights = {dataset: roots[dataset] / root_total for dataset in _DATASETS}
    scale = max(input_counts[dataset] / raw_weights[dataset] for dataset in _DATASETS)
    counts = {
        dataset: max(
            input_counts[dataset],
            math.ceil(scale * raw_weights[dataset] - 1e-12),
        )
        for dataset in _DATASETS
    }
    total = sum(counts.values())
    return counts, {dataset: counts[dataset] / total for dataset in _DATASETS}


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
    invalid_gt_field = sorted(
        dataset
        for dataset in _DATASETS
        if grading_manifest.config_for(dataset).gt_field != "data_level"
    )
    if invalid_gt_field:
        errors.append(
            "grading gt_field must be data_level for: "
            + ", ".join(invalid_gt_field)
        )
    if task_config.metadata_fields != ("field_name", "table_name"):
        errors.append(
            "task config metadata_fields must be field_name+table_name "
            "(contract change 2025-08-25)"
        )
    if source_report.get("family") != "rl-cascade":
        errors.append("release family must be rl-cascade")
    if source_report.get("format") != "dataclassify-finance-shougang-mixture-v1":
        errors.append("release report format must be the approved finance+shougang mixture")
    release = source_report.get("release")
    if not isinstance(release, Mapping) or release.get("status") != "passed" or release.get("published") is not True:
        errors.append("release must be passed and published")
    sampling = source_report.get("sampling")
    if not isinstance(sampling, Mapping) or sampling.get("policy") != "p(dataset) proportional to sqrt(source_count)":
        errors.append("release must record the approved sqrt sampling policy")
    else:
        input_counts = sampling.get("train_input_source_counts")
        source_counts = sampling.get("train_source_counts")
        achieved_weights = sampling.get("train_achieved_weights")
        if (
            not isinstance(input_counts, Mapping)
            or not isinstance(source_counts, Mapping)
            or not isinstance(achieved_weights, Mapping)
            or set(input_counts) != set(_DATASETS)
            or set(source_counts) != set(_DATASETS)
            or set(achieved_weights) != set(_DATASETS)
        ):
            errors.append("sqrt sampling report must contain both dataset count/weight maps")
        else:
            normalized_input: dict[str, int] = {}
            for dataset in _DATASETS:
                value = input_counts.get(dataset)
                if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                    errors.append(f"sqrt sampling input count for {dataset} is invalid")
                else:
                    normalized_input[dataset] = value
            if len(normalized_input) == len(_DATASETS):
                try:
                    expected_counts, expected_achieved = expected_sqrt_materialization(
                        normalized_input
                    )
                except ValueError:
                    errors.append("sqrt sampling input counts are invalid")
                    expected_counts, expected_achieved = {}, {}
                for dataset in _DATASETS:
                    raw_weight = achieved_weights.get(dataset)
                    if (
                        isinstance(raw_weight, bool)
                        or not isinstance(raw_weight, (int, float))
                        or not math.isfinite(float(raw_weight))
                        or abs(float(raw_weight) - expected_achieved[dataset]) > 1e-12
                    ):
                        errors.append(f"sqrt sampling achieved weight mismatch for {dataset}")
                    if source_counts.get(dataset) != expected_counts[dataset]:
                        errors.append(f"sqrt sampling source count mismatch for {dataset}")
    gate = source_report.get("label_gap_gate")
    if not isinstance(gate, Mapping) or gate.get("status") != "passed":
        errors.append("cascade release label/level gap gate must be passed")
    elif any(gate.get(name) for name in ("blocking", "blocking_levels", "waived", "waived_levels")):
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
            if not isinstance(manifest_datasets, Mapping) or set(manifest_datasets) != set(_DATASETS):
                errors.append("grading manifest lineage must contain both datasets")
            else:
                for dataset in _DATASETS:
                    details = manifest_datasets.get(dataset)
                    expected = details.get("sha256") if isinstance(details, Mapping) else None
                    if expected != grading_manifest.sha256_for(dataset):
                        errors.append(f"{dataset} grading standard hash mismatch")

    raw_inputs = source_report.get("inputs")
    if not isinstance(raw_inputs, Mapping) or set(raw_inputs) != set(_DATASETS):
        errors.append("cascade release must carry both input release lineages")
        raw_inputs = {}
    else:
        for dataset in _DATASETS:
            lineage = raw_inputs.get(dataset)
            if not isinstance(lineage, Mapping):
                errors.append(f"{dataset} input release lineage is malformed")
                continue
            report_path_raw = lineage.get("export_report_path")
            report_sha = lineage.get("export_report_sha256")
            parquet_paths = lineage.get("parquet_paths")
            parquet_hashes = lineage.get("parquet_sha256")
            if not isinstance(report_path_raw, str) or not isinstance(report_sha, str):
                errors.append(f"{dataset} input release report hash lineage is missing")
            else:
                report_file = Path(report_path_raw)
                if not report_file.is_file() or sha256_file(report_file) != report_sha:
                    errors.append(f"{dataset} input release report hash mismatch")
            if not isinstance(parquet_paths, Mapping) or not isinstance(parquet_hashes, Mapping):
                errors.append(f"{dataset} input parquet hash lineage is missing")
            else:
                for split in _SPLITS:
                    parquet_raw = parquet_paths.get(split)
                    expected = parquet_hashes.get(split)
                    if not isinstance(parquet_raw, str) or not isinstance(expected, str):
                        errors.append(f"{dataset} input {split} hash lineage is malformed")
                        continue
                    parquet_file = Path(parquet_raw)
                    if not parquet_file.is_file() or sha256_file(parquet_file) != expected:
                        errors.append(f"{dataset} input {split} parquet hash mismatch")
    raw_splits = source_report.get("splits")
    if not isinstance(raw_splits, Mapping):
        errors.append("release report must contain splits")
        raw_splits = {}

    all_ids: dict[str, set[str]] = {}
    datasets_seen: set[str] = set()
    stage2_rows = 0
    duplicate_source_ids = 0
    rows_by_split: dict[str, int] = {}
    rows_by_dataset: dict[str, dict[str, int]] = {
        split: {dataset: 0 for dataset in _DATASETS} for split in _SPLITS
    }
    for split in _SPLITS:
        parquet = root / f"{split}.parquet"
        details = raw_splits.get(split)
        if not parquet.is_file() or not isinstance(details, Mapping):
            errors.append(f"{split} artifact/report is missing")
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
            if dataset not in _DATASETS:
                errors.append(f"{split} row {index} has invalid dataset")
                continue
            split_datasets.add(dataset)
            datasets_seen.add(dataset)
            rows_by_dataset[split][dataset] += 1
            if stage != "stage1":
                stage2_rows += 1
                errors.append(f"{split} row {index} is not Stage1-only")
            if isinstance(source_id, str) and source_id:
                ids.append(source_id)
            row_errors = validate_rl_row(
                row,
                dataset=dataset,
                registry=registry,
                task_config=task_config,
                corpus=corpus,
                grading=grading_manifest.config_for(dataset),
            )
            if row_errors:
                errors.append(
                    f"{split} row {index} violates {len(row_errors)} RL contract checks"
                )
        duplicate_source_ids += len(ids) - len(set(ids))
        if len(ids) != len(set(ids)):
            errors.append(f"{split} contains duplicate source ids")
        if split_datasets != set(_DATASETS):
            errors.append(f"{split} mixture must contain finance and shougang")
        all_ids[split] = set(ids)
    cross_split_overlap = 0
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = all_ids.get(left, set()) & all_ids.get(right, set())
        cross_split_overlap += len(overlap)
    if cross_split_overlap:
        errors.append("source ids overlap across splits")

    # The report's source counts are part of the sampling contract, not an
    # unchecked annotation.  Compare them with the actual Stage1 episode
    # rows so a forged sqrt-mixture report cannot pass validation.
    for split in _SPLITS:
        details = raw_splits.get(split)
        reported_counts = details.get("source_counts") if isinstance(details, Mapping) else None
        if not isinstance(reported_counts, Mapping):
            errors.append(f"{split} report has no source_counts")
            continue
        for dataset in _DATASETS:
            expected = reported_counts.get(dataset)
            actual = rows_by_dataset[split][dataset]
            if isinstance(expected, bool) or not isinstance(expected, int) or expected != actual:
                errors.append(f"{split} source count mismatch for {dataset}")

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


def _args(argv: list[str] | None) -> argparse.Namespace:
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
