"""Audit whether field-only prompts deterministically identify leaf+data_level.

Only aggregate counts and SHA-256 values are written. Raw field names, target
labels, and source ids never appear in the report or console output.  The
``--bundle`` mode audits finance and shougang separately before producing one
verified aggregate suitable for reference-checkpoint provenance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from agent.hashing import sha256_file
from agent.training.input_audit import (
    audit_prompt_target_bundle,
    audit_prompt_target_conflicts,
)

_DATASETS = ("finance", "shougang")


def _args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--canonical", action="append", default=[], required=False,
        help=(
            "Canonical schema-v2 all.json. For --bundle use DATASET=PATH "
            "twice; single-dataset mode accepts one plain path."
        ),
    )
    parser.add_argument(
        "--dataset", action="append", default=[], metavar="DATASET=PATH",
        help="Alias for a DATASET=PATH canonical input in bundle mode",
    )
    parser.add_argument(
        "--classification-standard", action="append", required=True,
        help=(
            "Exact registry/corpus standard artifact. In bundle mode use "
            "DATASET=PATH once per dataset."
        ),
    )
    parser.add_argument(
        "--grading-standard", action="append", required=True,
        help=(
            "Exact approved grading rubric. In bundle mode use DATASET=PATH "
            "once per dataset."
        ),
    )
    parser.add_argument(
        "--bundle", action="store_true",
        help="Emit a verified finance+shougang audit bundle",
    )
    parser.add_argument("--split", default=None)
    parser.add_argument("--level-field", default="data_level")
    parser.add_argument("--report", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _named_paths(values: list[str], label: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{label} bundle entries must be DATASET=PATH")
        dataset, raw_path = value.split("=", 1)
        dataset = dataset.strip()
        if dataset not in _DATASETS or dataset in result or not raw_path.strip():
            raise ValueError(
                f"{label} bundle entries must cover finance and shougang exactly once"
            )
        result[dataset] = Path(raw_path).expanduser().resolve()
    if set(result) != set(_DATASETS):
        raise ValueError(
            f"{label} bundle entries must cover finance and shougang exactly once"
        )
    return result


def _load_records(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError("canonical input must be an array of objects")
    return value


def _write_report(output: Path, report: dict[str, Any]) -> None:
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


def main(argv: list[str] | None = None) -> int:
    args = _args(argv)
    output = Path(args.report)
    if output.exists() and not args.overwrite:
        print("status:error report_exists", file=sys.stderr)
        return 2
    try:
        if args.bundle:
            canonical_values = [*args.canonical, *args.dataset]
            canonical = _named_paths(canonical_values, "canonical")
            classification = _named_paths(
                args.classification_standard, "classification standard"
            )
            grading = _named_paths(args.grading_standard, "grading standard")
            report = audit_prompt_target_bundle(
                {dataset: _load_records(path) for dataset, path in canonical.items()},
                classification_standard_sha256_by_dataset={
                    dataset: sha256_file(path)
                    for dataset, path in classification.items()
                },
                grading_standard_sha256_by_dataset={
                    dataset: sha256_file(path) for dataset, path in grading.items()
                },
                split=args.split if args.split is not None else "all",
                level_field=args.level_field,
            )
        else:
            if args.dataset:
                raise ValueError("--dataset requires --bundle")
            if (
                len(args.canonical) != 1
                or len(args.classification_standard) != 1
                or len(args.grading_standard) != 1
            ):
                raise ValueError(
                    "single-dataset audit accepts one canonical and two standards"
                )
            value = _load_records(Path(args.canonical[0]).expanduser().resolve())
            report = audit_prompt_target_conflicts(
                value,
                classification_standard_sha256=sha256_file(
                    args.classification_standard[0]
                ),
                grading_standard_sha256=sha256_file(args.grading_standard[0]),
                split=args.split if args.split is not None else "train",
                level_field=args.level_field,
            )
        _write_report(output, report)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        # Never echo offending canonical values or paths into an audit log.
        print(f"status:error type={type(exc).__name__}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": report["status"],
                "records_audited": report["records_audited"],
                "conflict_keys": report["conflict_keys"],
                **({"datasets": list(_DATASETS)} if args.bundle else {}),
            },
            separators=(",", ":"),
        )
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
