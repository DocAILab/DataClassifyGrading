"""Publish a deterministic finance+shougang sqrt-weighted VeRL release.

Inputs must be separately validated, passed releases. Training uses the
accepted ``p(dataset) ∝ sqrt(source_count)`` policy. Val/test files are pooled
only for trainer diagnostics; formal EM/F1 remains per-dataset and is never
computed from these pooled files.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Mapping

from agent.hashing import sha256_file
from agent.task import GradingConfig, LeafRegistry, TaskConfig
from agent.task.assets import load_corpus_categories
from agent.task.grading_manifest import DatasetGradingManifest
from agent.task.prompt_choices import PromptChoiceRegistry
from agent.task.prompts import (
    build_stage1_prompt,
    build_stage2_prompt,
    stage1_answer,
    stage2_answer,
)
from agent.training.common import build_candidates, require_corpus_covers_registry
from agent.training.mixture import build_sqrt_mixture
from agent.training.rl import validate_rl_dataset
from agent.training.sft import validate_sft_dataset

_SPLITS = ("train", "val", "test")
_DATASETS = ("finance", "shougang")


def _args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("sft", "rl-cascade"), required=True)
    parser.add_argument(
        "--input", action="append", required=True, metavar="DATASET=DIR",
        help="Exactly finance=<passed release> and shougang=<passed release>",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--grading-manifest",
        help="Optional verified per-dataset grading manifest for provenance",
    )
    parser.add_argument("--registry", help="Leaf registry for output row validation")
    parser.add_argument("--corpus", help="Canonical corpus for output row validation")
    parser.add_argument("--metadata-fields", nargs="+", help="Prompt-visible fields for output validation")
    parser.add_argument("--task-config", help="Optional task config for output validation")
    parser.add_argument("--grading-config", help="Optional joint grading config for output validation")
    return parser.parse_args(argv)


def _inputs(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--input must be DATASET=DIR")
        name, raw_path = value.split("=", 1)
        name = name.strip()
        if name in result:
            raise ValueError(f"duplicate mixture input: {name}")
        result[name] = Path(raw_path).expanduser().resolve()
    if set(result) != set(_DATASETS):
        raise ValueError("mixture requires exactly finance and shougang inputs")
    return result


def _resolve_report_output(root: Path, report_path: Path, raw_path: object) -> Path:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError("mixture input split output_file must be a non-empty path")
    supplied = Path(raw_path)
    candidates = (
        [supplied]
        if supplied.is_absolute()
        else [root / supplied, report_path.parent / supplied, supplied]
    )
    expected_root = root.resolve()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved.is_file() and resolved.parent == expected_root:
            return resolved
    raise ValueError("mixture input split output_file is not the release artifact")


def _read_passed_release(
    root: Path,
    *,
    family: str,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("mixture build requires pyarrow") from exc
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"mixture input release not found: {root}")
    report_path = root / "export_report.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"input export report not found: {report_path}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("input export report contains duplicate keys")
            result[key] = value
        return result

    report = json.loads(
        report_path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates
    )
    if not isinstance(report, Mapping):
        raise ValueError("input export report must be an object")
    release = report.get("release")
    if not isinstance(release, Mapping) or release.get("status") != "passed" or release.get("published") is not True:
        raise ValueError("mixture input must be a passed published release")
    validation = report.get("validation")
    if not isinstance(validation, Mapping) or validation.get("valid") is not True:
        raise ValueError("mixture input validation.valid must be true")
    gate = report.get("label_gap_gate")
    if not isinstance(gate, Mapping) or gate.get("status") != "passed":
        raise ValueError("mixture input label/level gap gate must be passed without waiver")
    for name in ("blocking", "blocking_levels", "waived", "waived_levels"):
        values = gate.get(name, [])
        if values is None:
            values = []
        if not isinstance(values, list):
            raise ValueError(f"mixture input label gap field {name} must be an array")
        if values:
            raise ValueError("mixture input label/level gap gate must have no waivers")
    if family in {"sft", "rl-cascade"}:
        grading = report.get("grading")
        if not isinstance(grading, Mapping) or grading.get("enabled") is not True:
            raise ValueError("formal mixture inputs require joint grading")
        levels = grading.get("levels")
        if not isinstance(levels, list) or not levels or not all(
            isinstance(level, str) and level.strip() for level in levels
        ):
            raise ValueError("formal mixture inputs require grading levels")
    report_splits = report.get("splits")
    if not isinstance(report_splits, Mapping):
        raise ValueError("mixture input report has no splits")
    rows: dict[str, list[dict[str, Any]]] = {}
    split_hashes: dict[str, str] = {}
    parquet_paths: dict[str, str] = {}
    source_ids_by_split: dict[str, set[str]] = {}
    for split in _SPLITS:
        parquet = root / f"{split}.parquet"
        details = report_splits.get(split)
        if not parquet.is_file() or not isinstance(details, Mapping):
            raise ValueError(f"mixture input has no verified {split} parquet")
        reported_path = _resolve_report_output(root, report_path, details.get("output_file"))
        if reported_path != parquet.resolve():
            raise ValueError(f"mixture input {split} report points at the wrong artifact")
        actual = sha256_file(parquet)
        expected = details.get("parquet_sha256")
        if not isinstance(expected, str) or expected != actual:
            raise ValueError(f"mixture input {split} parquet hash mismatch")
        values = pq.read_table(parquet).to_pylist()
        if not values:
            raise ValueError(f"mixture input {split} parquet is empty")
        ids: list[str] = []
        stages_by_id: dict[str, list[str]] = {}
        for index, row in enumerate(values):
            if not isinstance(row, Mapping):
                raise ValueError(f"mixture input {split} row {index} is not an object")
            if family == "sft":
                source_id = row.get("source_id")
                stage = row.get("stage")
            else:
                extra = row.get("extra_info")
                source_id = extra.get("source_id") if isinstance(extra, Mapping) else None
                stage = extra.get("stage") if isinstance(extra, Mapping) else None
            if not isinstance(source_id, str) or not source_id.strip():
                raise ValueError(f"mixture input {split} row {index} has no source id")
            if stage not in {"stage1", "stage2"}:
                raise ValueError(f"mixture input {split} row {index} has an invalid stage")
            normalized_id = source_id.strip()
            ids.append(normalized_id)
            stages_by_id.setdefault(normalized_id, []).append(stage)
        source_ids = set(ids)
        for source_id, stages in stages_by_id.items():
            if sorted(stages) != ["stage1", "stage2"]:
                raise ValueError(
                    f"mixture input {split} source does not contain one stage1/stage2 pair"
                )
        if source_ids & set().union(*(source_ids_by_split.values())):
            raise ValueError("mixture input source ids overlap across splits")
        source_ids_by_split[split] = source_ids
        rows[split] = values
        split_hashes[split] = actual
        parquet_paths[split] = parquet.resolve().as_posix()
    return rows, {
        "release_dir": root.as_posix(),
        "export_report_path": report_path.resolve().as_posix(),
        "export_report_sha256": sha256_file(report_path),
        "parquet_paths": parquet_paths,
        "parquet_sha256": split_hashes,
        "format": report.get("format"),
        "family": report.get("family"),
        "grading": report.get("grading"),
        "validation": {"valid": True},
    }


def _load_output_contract(args: argparse.Namespace) -> tuple[LeafRegistry, TaskConfig, dict[str, Any], GradingConfig | None] | None:
    supplied = (args.registry, args.corpus, args.metadata_fields, args.task_config, args.grading_config)
    if not any(value is not None for value in supplied):
        return None
    if not args.registry or not args.corpus or not args.metadata_fields:
        raise ValueError(
            "output validation requires --registry, --corpus, and --metadata-fields"
        )
    config_data: dict[str, Any] = {}
    if args.task_config:
        loaded = json.loads(Path(args.task_config).read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("task config must be a JSON object")
        config_data = loaded
    config_data["metadata_fields"] = args.metadata_fields
    registry = LeafRegistry.from_path(args.registry)
    task = TaskConfig.from_mapping(config_data)
    if task.metadata_fields != ("field_name",):
        raise ValueError("formal mixture task metadata_fields must be exactly field_name")
    corpus = {
        item.category_id: item for item in load_corpus_categories(args.corpus)
    }
    require_corpus_covers_registry(corpus, registry)
    grading = GradingConfig.from_path(args.grading_config) if args.grading_config else None
    return registry, task, corpus, grading


def _validate_source_release(
    root: Path,
    dataset: str,
    *,
    family: str,
    registry: LeafRegistry,
    task: TaskConfig,
    corpus: Mapping[str, Any],
    grading_manifest: DatasetGradingManifest,
) -> dict[str, Any]:
    grading = grading_manifest.config_for(dataset)
    if family == "sft":
        report = validate_sft_dataset(
            root, registry, task, corpus=corpus, grading=grading
        )
    else:
        report = validate_rl_dataset(
            root, dataset, registry, task, corpus=corpus, grading=grading
        )
    if not report.get("valid", False):
        raise ValueError(f"{dataset} input release row validation failed")
    return report


def _reseed_sft_row(
    row: Mapping[str, Any],
    mixture_id: str,
    *,
    registry: LeafRegistry,
    task: TaskConfig,
    corpus: Mapping[str, Any],
    grading: GradingConfig | None,
) -> dict[str, Any]:
    """Rebuild source-seeded SFT fields after assigning a replica id."""

    result = dict(row)
    stage = result.get("stage")
    ground_truth = result.get("ground_truth")
    metadata = result.get("metadata")
    if stage not in {"stage1", "stage2"}:
        raise ValueError("SFT mixture row has an invalid stage")
    if not isinstance(ground_truth, str) or ground_truth not in registry.ids:
        raise ValueError("SFT mixture row has an invalid ground_truth")
    if not isinstance(metadata, Mapping) or set(metadata) != set(task.metadata_fields):
        raise ValueError("SFT mixture row metadata does not match task config")
    candidates = build_candidates(ground_truth, registry, source_id=mixture_id)
    choices = PromptChoiceRegistry.from_registry(registry)
    visible = {field: metadata[field] for field in task.metadata_fields}
    if stage == "stage1":
        prompt = build_stage1_prompt(visible, registry, task, choices=choices)
        answer = stage1_answer(candidates, choices=choices)
    else:
        level = result.get("ground_truth_level")
        if grading is not None and (not isinstance(level, str) or level not in grading.levels):
            raise ValueError("SFT mixture row has an invalid grading level")
        prompt = build_stage2_prompt(
            visible,
            candidates,
            registry,
            task,
            corpus=corpus,
            choices=choices,
            grading=grading,
        )
        answer = stage2_answer(ground_truth, candidates, level=level)
    result["source_id"] = mixture_id
    result["candidates"] = candidates
    result["messages"] = [
        {"role": "system", "content": prompt.system},
        {"role": "user", "content": prompt.user},
        {"role": "assistant", "content": answer},
    ]
    return result


def _validate_sft_output(
    staging: Path,
    *,
    registry: LeafRegistry,
    task: TaskConfig,
    corpus: Mapping[str, Any],
    grading_manifest: DatasetGradingManifest,
) -> dict[str, Any]:
    """Validate each output dataset with its own immutable grading rubric."""

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("SFT mixture validation requires pyarrow") from exc
    reports: dict[str, Any] = {}
    for dataset in _DATASETS:
        dataset_root = staging / f".validate-{dataset}"
        dataset_root.mkdir()
        for split in _SPLITS:
            rows = pq.read_table(staging / f"{split}.parquet").to_pylist()
            selected = [
                row
                for row in rows
                if isinstance(row.get("mixture_provenance"), Mapping)
                and row["mixture_provenance"].get("dataset") == dataset
            ]
            if not selected:
                raise ValueError(f"SFT mixture output has no {dataset} rows in {split}")
            pq.write_table(pa.Table.from_pylist(selected), dataset_root / f"{split}.parquet")
        report = validate_sft_dataset(
            dataset_root,
            registry,
            task,
            corpus=corpus,
            grading=grading_manifest.config_for(dataset),
        )
        reports[dataset] = report
        shutil.rmtree(dataset_root, ignore_errors=True)
        if not report.get("valid", False):
            raise ValueError(f"SFT mixture output validation failed for {dataset}")
    return {
        "format": "dataclassify-sft-mixture-validation-v1",
        "valid": True,
        "datasets": reports,
    }


def main(argv: list[str] | None = None) -> int:
    args = _args(argv)
    output = Path(args.output_dir).expanduser().resolve()
    staging: Path | None = None
    try:
        inputs = _inputs(args.input)
        if output.exists():
            raise FileExistsError("mixture output must be a new directory")
        if not args.grading_manifest:
            raise ValueError("mixture build requires --grading-manifest")
        grading_manifest = DatasetGradingManifest.from_path(args.grading_manifest)
        if not args.registry or not args.corpus or not args.metadata_fields:
            raise ValueError(
                "mixture build requires --registry, --corpus, and --metadata-fields"
            )
        output_contract = _load_output_contract(args)
        rows_by_dataset: dict[str, dict[str, list[dict[str, Any]]]] = {}
        input_lineage: dict[str, Any] = {}
        registry, task, corpus, _ = output_contract
        for dataset in _DATASETS:
            rows_by_dataset[dataset], input_lineage[dataset] = _read_passed_release(
                inputs[dataset], family=args.family
            )
            source_grading = input_lineage[dataset].get("grading")
            expected_grading = grading_manifest.config_for(dataset)
            if not isinstance(source_grading, Mapping):
                raise ValueError(f"{dataset} input release has no grading asset metadata")
            if source_grading.get("levels") != list(expected_grading.levels) or source_grading.get("gt_field") != expected_grading.gt_field:
                raise ValueError(f"{dataset} input grading asset does not match manifest")
            reported_grading_sha = source_grading.get("standard_sha256")
            if not isinstance(reported_grading_sha, str):
                raise ValueError(f"{dataset} input grading asset hash is missing")
            if reported_grading_sha != grading_manifest.sha256_for(dataset):
                raise ValueError(f"{dataset} input grading asset hash mismatch")
            source_validation = _validate_source_release(
                inputs[dataset],
                dataset,
                family=args.family,
                registry=registry,
                task=task,
                corpus=corpus,
                grading_manifest=grading_manifest,
            )
            input_lineage[dataset]["validation"] = {
                "valid": True,
                "format": source_validation.get("format"),
            }
        output.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent)
        )
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("mixture build requires pyarrow") from exc

        split_reports: dict[str, Any] = {}
        results = {}
        for split in _SPLITS:
            result = build_sqrt_mixture(
                {
                    dataset: rows_by_dataset[dataset][split]
                    for dataset in _DATASETS
                },
                family=args.family,
                split=split,
            )
            results[split] = result
            output_rows = list(result.rows)
            if args.family == "sft" and output_contract is not None:
                registry, task, corpus, _ = output_contract
                output_rows = [
                    _reseed_sft_row(
                        row,
                        row["source_id"],
                        registry=registry,
                        task=task,
                        corpus=corpus,
                        grading=(
                            grading_manifest.config_for(
                                row["mixture_provenance"]["dataset"]
                            )
                            if isinstance(row.get("mixture_provenance"), Mapping)
                            else grading_manifest.config_for("finance")
                        ),
                    )
                    for row in output_rows
                ]
            parquet = staging / f"{split}.parquet"
            pq.write_table(pa.Table.from_pylist(output_rows), parquet)
            split_reports[split] = {
                "output_file": (output / parquet.name).as_posix(),
                "parquet_sha256": sha256_file(parquet),
                "rows": len(output_rows),
                "source_counts": result.source_counts,
                "input_source_counts": result.input_source_counts,
                "achieved_weights": result.achieved_weights,
            }
        train = results["train"]
        manifest_lineage = None
        if grading_manifest is not None:
            manifest_lineage = {
                "path": grading_manifest.source_path.as_posix(),
                "sha256": sha256_file(grading_manifest.source_path),
                "datasets": {
                    dataset: {"sha256": grading_manifest.sha256_for(dataset)}
                    for dataset in _DATASETS
                },
            }
        registry, task, corpus, grading = output_contract
        if args.family == "sft" and grading is None:
            grading = grading_manifest.config_for("finance")
        validation_report: dict[str, Any]
        if args.family == "sft":
            validation_report = _validate_sft_output(
                staging,
                registry=registry,
                task=task,
                corpus=corpus,
                grading_manifest=grading_manifest,
            )
        else:
            if grading_manifest is None:
                raise ValueError("formal RL mixture requires grading manifest")
            validation_report = {"valid": True, "format": "pending-cascade-validation-v1"}
        report = {
            "format": "dataclassify-finance-shougang-mixture-v1",
            "family": args.family,
            "release": {"status": "passed", "published": True},
            "label_gap_gate": {
                "status": "passed",
                "blocking": [],
                "blocking_levels": [],
                "waived": [],
                "waived_levels": [],
            },
            "validation": validation_report,
            "sampling": {
                "policy": "p(dataset) proportional to sqrt(source_count)",
                "train_input_source_counts": train.input_source_counts,
                "train_source_counts": train.source_counts,
                "train_achieved_weights": train.achieved_weights,
            },
            "inputs": input_lineage,
            "official_evaluation": "per-dataset val/test releases only",
            **({"grading_manifest": manifest_lineage} if manifest_lineage else {}),
            "splits": split_reports,
        }
        (staging / "export_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if args.family == "rl-cascade":
            from script.verl.rl.validate_cascade import validate_cascade_release

            validation_report = validate_cascade_release(
                staging,
                registry=registry,
                task_config=task,
                corpus=corpus,
                grading_manifest=grading_manifest,
            )
            if not validation_report.get("valid", False):
                raise ValueError("cascade mixture output validation failed")
            report["validation"] = validation_report
            (staging / "export_report.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        os.replace(staging, output)
        staging = None
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"build_mixture: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    finally:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
    print(
        json.dumps(
            {
                "status": "passed",
                "family": args.family,
                "output_dir": output.as_posix(),
            },
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
