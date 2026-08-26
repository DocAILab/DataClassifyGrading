"""Summarize formal shougang EM and composite pair Macro-F1."""

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
    FORMAL_SAMPLING_POLICY,
)


def _metric(value: Any, name: str, dataset: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{dataset} {name} must be numeric")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{dataset} {name} must be within [0, 1]")
    return result


def aggregate_reports(paths: Mapping[str, str | Path]) -> dict[str, Any]:
    """Build a singleton shougang metric summary.

    The mapping interface and ``datasets``/``overall`` shape are retained for
    callers of the former joint aggregator, but exactly one formal report is
    accepted and no cross-dataset averaging is performed.
    """

    if not isinstance(paths, Mapping) or set(paths) != FORMAL_DATASET_SET:
        raise ValueError(
            "metric inputs must be exactly the formal dataset set: "
            + ", ".join(FORMAL_DATASETS)
        )
    datasets: dict[str, Any] = {}
    for dataset in FORMAL_DATASETS:
        path = Path(paths[dataset])
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise ValueError(f"{dataset} metric report must be an object")
        reported_dataset = value.get("dataset")
        if reported_dataset is not None and reported_dataset != dataset:
            raise ValueError(
                f"{dataset} metric report identifies a different dataset"
            )
        metrics = value.get("metrics")
        if not isinstance(metrics, Mapping):
            raise ValueError(f"{dataset} evaluator report must contain metrics")
        em = _metric(metrics.get("strict_joint_em"), "strict_joint_em", dataset)
        f1 = _metric(
            metrics.get("composite_macro_f1"), "composite_macro_f1", dataset
        )
        datasets[dataset] = {
            "joint_em": em,
            "composite_macro_f1": f1,
            "report_path": path.as_posix(),
            "report_sha256": sha256_file(path),
        }
    dataset = FORMAL_DATASETS[0]
    summary = datasets[dataset]
    return {
        "format": "dataclassify-shougang-metrics-v1",
        "release_format": FORMAL_RELEASE_FORMAT,
        "dataset": dataset,
        "datasets": datasets,
        "overall": {
            "joint_em": summary["joint_em"],
            "composite_macro_f1": summary["composite_macro_f1"],
            "aggregation": FORMAL_SAMPLING_POLICY,
        },
    }


def _args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, metavar="DATASET=REPORT")
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _parse_inputs(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--input must be DATASET=REPORT")
        dataset, raw_path = value.split("=", 1)
        dataset = dataset.strip()
        if dataset in result:
            raise ValueError(f"duplicate metric input: {dataset}")
        result[dataset] = Path(raw_path)
    return result


def main(argv: list[str] | None = None) -> int:
    args = _args(argv)
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        print("aggregate_joint_metrics: output exists", file=sys.stderr)
        return 2
    try:
        report = aggregate_reports(_parse_inputs(args.input))
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.tmp")
        try:
            temporary.write_text(
                json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            temporary.replace(output)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"aggregate_joint_metrics: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report["overall"], separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
