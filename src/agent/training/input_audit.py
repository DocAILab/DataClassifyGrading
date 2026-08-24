"""Prompt-identifiability audit for the field-only joint task.

When the model sees only ``field_name`` plus immutable standards, identical
visible inputs cannot legitimately own different ``(leaf, data_level)``
targets.  Reports intentionally contain hashes and counts only: no raw field,
target, or record-id values are emitted.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_WS_RE = re.compile(r"\s+")
_BUNDLE_DATASETS = ("finance", "shougang")


def normalize_field_name(value: str) -> str:
    """Normalize prompt-equivalent field identifiers for collision checks."""

    if not isinstance(value, str):
        raise ValueError("field_name must be a string")
    normalized = unicodedata.normalize("NFKC", value)
    normalized = _WS_RE.sub(" ", normalized.strip()).casefold()
    if not normalized:
        raise ValueError("field_name must be non-empty")
    return normalized


def _digest(*parts: str) -> str:
    value = "\0".join(parts).encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def _standard_sha(value: str, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a lowercase 64-character SHA-256")
    normalized = value.lower()
    if not _SHA256_RE.fullmatch(normalized):
        raise ValueError(f"{name} must be a lowercase 64-character SHA-256")
    return normalized


def audit_prompt_target_conflicts(
    records: Iterable[Mapping[str, Any]],
    *,
    classification_standard_sha256: str,
    grading_standard_sha256: str,
    split: str = "train",
    level_field: str = "data_level",
    dataset: str | None = None,
) -> dict[str, Any]:
    """Return a deterministic, redacted identifiability report."""

    if dataset is not None and dataset not in _BUNDLE_DATASETS:
        raise ValueError("dataset must be finance or shougang")
    classification_sha = _standard_sha(
        classification_standard_sha256, "classification_standard_sha256"
    )
    grading_sha = _standard_sha(
        grading_standard_sha256, "grading_standard_sha256"
    )
    grouped_targets: dict[str, set[str]] = defaultdict(set)
    grouped_records: dict[str, list[str]] = defaultdict(list)
    audited = 0
    if split is not None and (not isinstance(split, str) or not split.strip()):
        raise ValueError("split must be a non-empty string or None")
    audit_all_splits = split is None or split.strip().casefold() in {"all", "*"}
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"record {index} must be a mapping")
        if str(record.get("resolution_status", "")).strip() != "resolved":
            continue
        if not audit_all_splits and str(record.get("split", "")).strip() != split:
            continue
        metadata = record.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError(f"record {index} has no metadata mapping")
        field_name = normalize_field_name(metadata.get("field_name"))
        target = record.get("target")
        if not isinstance(target, Mapping):
            raise ValueError(f"record {index} has no target mapping")
        leaf = target.get("category_id")
        level = record.get(level_field)
        record_id = record.get("id")
        if not isinstance(leaf, str) or not leaf.strip():
            raise ValueError(f"record {index} has no target.category_id")
        if not isinstance(level, str) or not level.strip():
            raise ValueError(f"record {index} has no {level_field}")
        if not isinstance(record_id, str) or not record_id.strip():
            raise ValueError(f"record {index} has no stable id")
        key_sha = _digest(field_name, classification_sha, grading_sha)
        target_sha = _digest(leaf.strip(), level.strip())
        grouped_targets[key_sha].add(target_sha)
        grouped_records[key_sha].append(_digest(record_id.strip()))
        audited += 1

    conflicts: list[dict[str, Any]] = []
    for key_sha in sorted(grouped_targets):
        targets = sorted(grouped_targets[key_sha])
        if len(targets) <= 1:
            continue
        record_hashes = sorted(grouped_records[key_sha])
        conflicts.append(
            {
                "prompt_key_sha256": key_sha,
                "record_count": len(record_hashes),
                "record_id_sha256": record_hashes,
                "target_count": len(targets),
                "target_sha256": targets,
            }
        )
    all_keys = sorted(grouped_targets)
    report = {
        **({"dataset": dataset} if dataset is not None else {}),
        "format": "field-prompt-identifiability-v1",
        "status": "failed" if conflicts else "passed",
        "split": "all" if audit_all_splits else split,
        "classification_standard_sha256": classification_sha,
        "grading_standard_sha256": grading_sha,
        "records_audited": audited,
        "prompt_keys": len(all_keys),
        "prompt_key_sha256": _digest(*all_keys),
        "conflict_keys": len(conflicts),
        "conflicting_records": sum(item["record_count"] for item in conflicts),
        "conflicts": conflicts,
    }
    return report


def require_identifiable_prompts(report: Mapping[str, Any]) -> None:
    """Fail closed unless an audit report proves deterministic targets."""

    if not isinstance(report, Mapping):
        raise ValueError("unsupported prompt-identifiability report")
    if report.get("format") != "field-prompt-identifiability-v1":
        raise ValueError("unsupported prompt-identifiability report")
    if report.get("status") != "passed" or report.get("conflict_keys") != 0:
        raise ValueError(
            "prompt-identifiability failed: field-only prompts are not identifiable from their joint targets"
        )
    conflicts = report.get("conflicts")
    if conflicts not in (None, []):
        raise ValueError("prompt-identifiability report contains conflicts")


def _bundle_standard_maps(
    datasets: set[str],
    *,
    standards_by_dataset: Mapping[str, Any] | None,
    classification_standard_sha256_by_dataset: Mapping[str, str] | None,
    grading_standard_sha256_by_dataset: Mapping[str, str] | None,
    classification_standard_sha256: Mapping[str, str] | None,
    grading_standard_sha256: Mapping[str, str] | None,
) -> tuple[dict[str, str], dict[str, str]]:
    """Normalize the supported bundle standard-map spellings.

    ``standards_by_dataset`` is the preferred shape for callers that keep
    both hashes together.  The explicit ``*_by_dataset`` arguments are kept
    as a convenient, unambiguous alternative for pipeline code.
    """

    if standards_by_dataset is not None and any(
        item is not None
        for item in (
            classification_standard_sha256_by_dataset,
            grading_standard_sha256_by_dataset,
            classification_standard_sha256,
            grading_standard_sha256,
        )
    ):
        raise ValueError("provide either standards_by_dataset or separate standard maps")
    if standards_by_dataset is not None:
        if set(standards_by_dataset) != datasets:
            raise ValueError("prompt-audit standards must cover exactly both datasets")
        classification: dict[str, str] = {}
        grading: dict[str, str] = {}
        for dataset in sorted(datasets):
            entry = standards_by_dataset[dataset]
            if isinstance(entry, Mapping):
                raw_classification = entry.get("classification_standard_sha256")
                if raw_classification is None:
                    raw_classification = entry.get("classification_sha256")
                raw_grading = entry.get("grading_standard_sha256")
                if raw_grading is None:
                    raw_grading = entry.get("grading_sha256")
            elif isinstance(entry, Sequence) and not isinstance(entry, (str, bytes)):
                if len(entry) != 2:
                    raise ValueError(
                        f"prompt-audit standards for {dataset!r} must contain two hashes"
                    )
                raw_classification, raw_grading = entry
            else:
                raise ValueError(
                    f"prompt-audit standards for {dataset!r} must be a mapping or pair"
                )
            if not isinstance(raw_classification, str) or not isinstance(raw_grading, str):
                raise ValueError(
                    f"prompt-audit standards for {dataset!r} require classification and grading hashes"
                )
            classification[dataset] = _standard_sha(
                raw_classification, f"{dataset} classification_standard_sha256"
            )
            grading[dataset] = _standard_sha(
                raw_grading, f"{dataset} grading_standard_sha256"
            )
        return classification, grading

    classification_map = (
        classification_standard_sha256_by_dataset
        if classification_standard_sha256_by_dataset is not None
        else classification_standard_sha256
    )
    grading_map = (
        grading_standard_sha256_by_dataset
        if grading_standard_sha256_by_dataset is not None
        else grading_standard_sha256
    )
    if classification_map is None or grading_map is None:
        raise ValueError("prompt-audit bundle requires per-dataset standard hashes")
    if set(classification_map) != datasets or set(grading_map) != datasets:
        raise ValueError("prompt-audit standard maps must cover exactly both datasets")
    return {
        dataset: _standard_sha(
            classification_map[dataset], f"{dataset} classification_standard_sha256"
        )
        for dataset in sorted(datasets)
    }, {
        dataset: _standard_sha(
            grading_map[dataset], f"{dataset} grading_standard_sha256"
        )
        for dataset in sorted(datasets)
    }


def audit_prompt_target_bundle(
    records_by_dataset: Mapping[str, Iterable[Mapping[str, Any]]],
    *,
    standards_by_dataset: Mapping[str, Any] | None = None,
    classification_standard_sha256_by_dataset: Mapping[str, str] | None = None,
    grading_standard_sha256_by_dataset: Mapping[str, str] | None = None,
    # Singular aliases make the call site parallel to the single-dataset API.
    classification_standard_sha256: Mapping[str, str] | None = None,
    grading_standard_sha256: Mapping[str, str] | None = None,
    split: str | None = "all",
    level_field: str = "data_level",
) -> dict[str, Any]:
    """Audit every dataset in the formal finance+shougang release.

    A single field-only audit is not enough for a joint release: standards and
    target vocabularies are dataset-local.  This function emits one redacted
    report per dataset and a deterministic aggregate.  No record, field,
    target, or raw standard value is copied into the returned mapping.
    """

    if not isinstance(records_by_dataset, Mapping):
        raise ValueError("prompt-audit records must be a dataset mapping")
    if split is not None and (not isinstance(split, str) or not split.strip()):
        raise ValueError("split must be a non-empty string or None")
    datasets = set(records_by_dataset)
    if datasets != set(_BUNDLE_DATASETS):
        raise ValueError("prompt-audit bundle requires exactly finance and shougang records")
    classification, grading = _bundle_standard_maps(
        datasets,
        standards_by_dataset=standards_by_dataset,
        classification_standard_sha256_by_dataset=classification_standard_sha256_by_dataset,
        grading_standard_sha256_by_dataset=grading_standard_sha256_by_dataset,
        classification_standard_sha256=classification_standard_sha256,
        grading_standard_sha256=grading_standard_sha256,
    )
    reports: dict[str, dict[str, Any]] = {}
    for dataset in _BUNDLE_DATASETS:
        report = audit_prompt_target_conflicts(
            records_by_dataset[dataset],
            classification_standard_sha256=classification[dataset],
            grading_standard_sha256=grading[dataset],
            split=split,
            level_field=level_field,
        )
        report["dataset"] = dataset
        reports[dataset] = report
    aggregate_prompt_keys = [
        f"{dataset}\0{reports[dataset]['prompt_key_sha256']}"
        for dataset in _BUNDLE_DATASETS
    ]
    return {
        "format": "field-prompt-identifiability-bundle-v1",
        "status": "passed"
        if all(report["status"] == "passed" for report in reports.values())
        else "failed",
        "split": "all" if split is None or split.strip().casefold() in {"all", "*"} else split,
        "datasets": reports,
        "standard_hashes": {
            dataset: {
                "classification_standard_sha256": classification[dataset],
                "grading_standard_sha256": grading[dataset],
            }
            for dataset in _BUNDLE_DATASETS
        },
        "records_audited": sum(report["records_audited"] for report in reports.values()),
        "prompt_keys": sum(report["prompt_keys"] for report in reports.values()),
        "prompt_key_sha256": _digest(*aggregate_prompt_keys),
        "conflict_keys": sum(report["conflict_keys"] for report in reports.values()),
        "conflicting_records": sum(
            report["conflicting_records"] for report in reports.values()
        ),
    }


# Descriptive aliases for callers that name the operation after its grouping.
audit_prompt_target_conflicts_by_dataset = audit_prompt_target_bundle
build_prompt_audit_bundle = audit_prompt_target_bundle


def require_identifiable_prompt_bundle(report: Mapping[str, Any]) -> None:
    """Fail closed unless both dataset-local audits are present and clean."""

    if not isinstance(report, Mapping):
        raise ValueError("unsupported prompt-identifiability bundle")
    expected_top_level = {
        "format", "status", "split", "datasets", "standard_hashes",
        "records_audited", "prompt_keys", "prompt_key_sha256", "conflict_keys",
        "conflicting_records",
    }
    if set(report) != expected_top_level:
        raise ValueError("prompt-identifiability bundle has unexpected fields")
    if report.get("format") != "field-prompt-identifiability-bundle-v1":
        raise ValueError("unsupported prompt-identifiability bundle")
    datasets = report.get("datasets")
    if not isinstance(datasets, Mapping) or set(datasets) != set(_BUNDLE_DATASETS):
        raise ValueError("prompt-identifiability bundle must contain finance and shougang audits")
    standards = report.get("standard_hashes")
    if not isinstance(standards, Mapping) or set(standards) != set(_BUNDLE_DATASETS):
        raise ValueError("prompt-identifiability bundle has incomplete standard hashes")
    total_records = 0
    total_keys = 0
    total_conflicts = 0
    total_conflicting_records = 0
    aggregate_prompt_keys: list[str] = []
    for dataset in _BUNDLE_DATASETS:
        item = datasets[dataset]
        if not isinstance(item, Mapping) or item.get("dataset") != dataset:
            raise ValueError(f"prompt-identifiability {dataset} audit is malformed")
        require_identifiable_prompts(item)
        expected_keys = {
            "format", "status", "split", "classification_standard_sha256",
            "grading_standard_sha256", "records_audited", "prompt_keys",
            "prompt_key_sha256", "conflict_keys", "conflicting_records", "conflicts",
            "dataset",
        }
        if set(item) != expected_keys:
            raise ValueError(f"prompt-identifiability {dataset} audit has unexpected fields")
        standard = standards[dataset]
        if not isinstance(standard, Mapping):
            raise ValueError(f"prompt-identifiability {dataset} standards are malformed")
        if set(standard) != {
            "classification_standard_sha256", "grading_standard_sha256"
        }:
            raise ValueError(f"prompt-identifiability {dataset} standards are malformed")
        classification_sha = _standard_sha(
            item["classification_standard_sha256"],
            f"{dataset} classification_standard_sha256",
        )
        grading_sha = _standard_sha(
            item["grading_standard_sha256"], f"{dataset} grading_standard_sha256"
        )
        if standard["classification_standard_sha256"] != classification_sha:
            raise ValueError(f"prompt-identifiability {dataset} classification hash mismatch")
        if standard["grading_standard_sha256"] != grading_sha:
            raise ValueError(f"prompt-identifiability {dataset} grading hash mismatch")
        conflicts = item.get("conflicts")
        if not isinstance(conflicts, list):
            raise ValueError(f"prompt-identifiability {dataset} conflicts are malformed")
        if any(not isinstance(conflict, Mapping) for conflict in conflicts):
            raise ValueError(f"prompt-identifiability {dataset} conflicts are malformed")
        if item.get("conflict_keys") != 0 or conflicts:
            raise ValueError(
                "prompt-identifiability failed: field-only prompts are not identifiable for every dataset"
            )
        for field in ("records_audited", "prompt_keys", "conflicting_records"):
            value = item.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"prompt-identifiability {dataset} {field} is invalid")
        total_records += item["records_audited"]
        total_keys += item["prompt_keys"]
        total_conflicts += item["conflict_keys"]
        total_conflicting_records += item["conflicting_records"]
        aggregate_prompt_keys.append(f"{dataset}\0{item['prompt_key_sha256']}")
    if report.get("status") != "passed":
        raise ValueError("prompt-identifiability bundle is not passed")
    if report.get("records_audited") != total_records:
        raise ValueError("prompt-identifiability bundle record count mismatch")
    if report.get("prompt_keys") != total_keys:
        raise ValueError("prompt-identifiability bundle prompt-key count mismatch")
    if report.get("conflict_keys") != total_conflicts or total_conflicts:
        raise ValueError("prompt-identifiability failed: bundle contains conflicts")
    if report.get("conflicting_records") != total_conflicting_records:
        raise ValueError("prompt-identifiability bundle conflict count mismatch")
    if report.get("prompt_key_sha256") != _digest(*aggregate_prompt_keys):
        raise ValueError("prompt-identifiability bundle prompt-key digest mismatch")


# Verification alias mirrors the builder's name for release-gate call sites.
verify_prompt_audit_bundle = require_identifiable_prompt_bundle


__all__ = [
    "normalize_field_name",
    "audit_prompt_target_conflicts",
    "require_identifiable_prompts",
    "audit_prompt_target_bundle",
    "audit_prompt_target_conflicts_by_dataset",
    "build_prompt_audit_bundle",
    "require_identifiable_prompt_bundle",
    "verify_prompt_audit_bundle",
]
