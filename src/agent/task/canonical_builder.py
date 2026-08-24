"""Canonical dataset construction: resolve every processed record against the
runtime-supplied LeafRegistry and emit the canonical contract (schema v2).

This module rebuilds the layer removed during the data-sanitation pass as
generic, runtime-configured code. All inputs are explicit arguments:

- processed records (list of dicts, e.g. ``<runtime>/processed/<ds>/all.json``);
- a runtime-loaded :class:`DatasetConfig` (the per-dataset manifest: leaf
  definition, id strategy, placeholders, projection exclusions, registry
  derivation) — nothing is built into source code;
- the LeafRegistry (the final constraint on the category universe);
- optional corpus categories (required for ``id_strategy="code"`` to build
  the leaf->code map).

Resolution outcomes stay explicit (see ``agent.task.resolver``); a resolver
RESOLVED result is downgraded to PATH_MISMATCH / MISSING_LEAF when the
generated category_id is not covered by the registry. Nothing is auto-fixed.

Output record shape (v2 = v1 superset, so existing consumers of
``resolution_status`` / ``target`` keep working):

    { ...original processed record fields, untouched...,
      "schema_version": 2,
      "dataset": <name>,
      "path_mask":   [bool, ...],           # which configured levels are non-empty
      "leaf":        {"field": ..., "value": ...},
      "resolution":  {"status": ..., "category_id": ...|null, "reason": ...},
      "resolution_status": <status>,         # v1 compat key
      "target": <canonical target mapping>,  # v1 compat key, resolved only
      "split": null,                         # populated by the split stage
      "split_exclusion_reason": null }

The report is deterministic (input sha256; no timestamps, no machine-local
paths). Paths are echoed exactly as given by the caller.
"""

from __future__ import annotations

import copy
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from agent.hashing import sha256_file
from agent.task.contracts import CorpusCategory, LeafRegistry, SampleTarget
from agent.task.dataset_config import DatasetConfig
from agent.task.identity import code_leaf_map
from agent.task.resolver import (
    ClassificationTargetResolver,
    ResolutionResult,
    ResolutionStatus,
)

SCHEMA_VERSION = 2


def _as_posix(path: str | Path) -> str:
    return Path(path).as_posix()


def load_processed_records(path: str | Path) -> list[Any]:
    """Load a processed all.json file (a JSON list). Fail-fast on shape."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"processed dataset not found: {source}")
    with source.open(encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError(f"{source} must contain a JSON list")
    return records


def load_corpus_categories_file(path: str | Path) -> list[CorpusCategory]:
    """Deserialize a canonical corpus JSON into CorpusCategory objects."""
    source = Path(path)
    with source.open(encoding="utf-8") as handle:
        value = json.load(handle)
    raw = value.get("categories") if isinstance(value, Mapping) else value
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError(f"corpus must contain a categories array: {source}")
    categories: list[CorpusCategory] = []
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise ValueError(f"invalid corpus category at index {index}: {source}")
        raw_code = item.get("code")
        categories.append(
            CorpusCategory(
                category_id=str(item.get("category_id", "")).strip(),
                name=str(item.get("name", "")).strip(),
                description=str(item.get("description", "") or ""),
                descriptions=tuple(str(part) for part in item.get("descriptions", ())),
                path=tuple(str(part) for part in item.get("path", ())),
                code=str(raw_code).strip() if raw_code is not None else None,
                examples=tuple(str(part) for part in item.get("examples", ())),
            )
        )
    ids = [category.category_id for category in categories]
    if len(ids) != len(set(ids)):
        raise ValueError(f"corpus category_id values must be unique: {source}")
    return categories


def resolve_record(
    record: Any,
    resolver: ClassificationTargetResolver,
    registry: LeafRegistry,
    registry_names: set[str],
    excluded_ids: frozenset[str] = frozenset(),
) -> ResolutionResult:
    """Resolve one record with the LeafRegistry as the final constraint.

    A resolver RESOLVED result whose category_id misses the registry is
    downgraded to PATH_MISMATCH (leaf known in the corpus/registry universe,
    identity path differs) or MISSING_LEAF (leaf absent from the universe).
    A resolved hit on a ``projection_excluded_category_ids`` entry is a
    configuration error (the registry/corpus build should have excluded it)
    and raises instead of training on it.
    """
    result = resolver.resolve_detailed(record)
    if result.status is not ResolutionStatus.RESOLVED or result.target is None:
        return result
    target = result.target
    if target.category_id in excluded_ids:
        raise ValueError(
            f"resolved target hit projection-excluded category "
            f"{target.category_id!r}: remove it from the registry/corpus or "
            "fix the config"
        )
    if target.category_id in registry.ids:
        return result
    if target.leaf_name in registry_names:
        return ResolutionResult(
            ResolutionStatus.PATH_MISMATCH,
            target=target,
            reason=(
                f"leaf {target.leaf_name!r} exists in the registry universe but "
                f"resolved identity {target.category_id!r} is not in the registry"
            ),
            leaf=target.leaf_name,
            category_id=target.category_id,
        )
    return ResolutionResult(
        ResolutionStatus.MISSING_LEAF,
        target=target,
        reason=f"leaf {target.leaf_name!r} is absent from the registry universe",
        leaf=target.leaf_name,
        category_id=target.category_id,
    )


def _classification_levels(record: Mapping[str, Any], config: DatasetConfig) -> list[str]:
    classification = record.get("classification")
    if not isinstance(classification, Mapping):
        return [""] * len(config.path_fields)
    return [
        str(classification.get(field, "") or "").strip()
        for field in config.path_fields
    ]


def _enrich_record(
    record: Any,
    dataset: str,
    config: DatasetConfig,
    result: ResolutionResult,
) -> dict[str, Any]:
    """Return the v2 canonical copy of one input record (input untouched)."""
    if isinstance(record, Mapping):
        canonical = copy.deepcopy(dict(record))
        levels = _classification_levels(record, config)
    else:
        # non-object record (e.g. a bare string): keep it auditable without
        # corrupting the canonical schema
        canonical = {"record": copy.deepcopy(record)}
        levels = [""] * len(config.path_fields)

    attempted_id = result.category_id or ""
    resolution_block = {
        "status": result.status.value,
        "category_id": attempted_id if attempted_id else None,
        "reason": result.reason,
    }
    if result.status is ResolutionStatus.RESOLVED:
        assert result.target is not None
        resolution_block["category_id"] = result.target.category_id

    canonical.update(
        {
            "schema_version": SCHEMA_VERSION,
            "dataset": dataset,
            "path_mask": [bool(level) for level in levels],
            "leaf": {
                "field": config.leaf_level,
                "value": levels[config.path_fields.index(config.leaf_level)],
            },
            "resolution": resolution_block,
            # v1 compatibility keys: existing consumers read these top-level
            # fields; they carry identical semantics to resolution/target.
            "resolution_status": result.status.value,
            "split": None,
            "split_exclusion_reason": None,
        }
    )
    if result.status is ResolutionStatus.RESOLVED:
        assert result.target is not None
        canonical["target"] = _target_mapping(result.target)
    else:
        # never leak a stale target from the input side
        canonical.pop("target", None)
    return canonical


def _target_mapping(target: SampleTarget) -> dict[str, Any]:
    return {
        "leaf_level": target.leaf_level,
        "leaf_name": target.leaf_name,
        "category_id": target.category_id,
        "category_path": list(target.category_path),
    }


@dataclass(frozen=True)
class CanonicalBuildResult:
    """Outcome of building one canonical dataset (pure computation result)."""

    dataset: str
    schema_version: int
    input_records: int
    output_records: int
    status_counts: Mapping[str, int]
    unresolved_details: Mapping[str, Any]
    resolved_targets_in_registry: bool
    registry_size: int
    registry_file: str
    registry_source: str | None
    registry_derivation: str
    input_file: str
    output_file: str
    input_sha256: str

    def to_mapping(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "schema_version": self.schema_version,
            "input_records": self.input_records,
            "output_records": self.output_records,
            "status_counts": dict(self.status_counts),
            "resolved_targets_in_registry": self.resolved_targets_in_registry,
            "registry_size": self.registry_size,
            "registry_file": self.registry_file,
            "registry_source": self.registry_source,
            "registry_derivation": self.registry_derivation,
            "input_file": self.input_file,
            "output_file": self.output_file,
            "input_sha256": self.input_sha256,
            "unresolved_details": dict(self.unresolved_details),
        }


def prepare_canonical_dataset(
    dataset: str,
    *,
    processed_file: str | Path,
    output_file: str | Path,
    config: DatasetConfig,
    registry: LeafRegistry,
    registry_file: str | Path = "",
    corpus_categories: Sequence[CorpusCategory] | None = None,
) -> tuple[CanonicalBuildResult, list[dict[str, Any]]]:
    """Load, validate and resolve one dataset's records. Pure computation.

    The caller decides when to write via :func:`write_canonical_dataset`,
    which enables cross-dataset fail-fast (no partial outputs when any
    selected dataset fails).
    """
    if config.dataset != dataset:
        raise ValueError(
            f"config dataset {config.dataset!r} does not match requested {dataset!r}"
        )

    registry_names = {category.name for category in registry.categories}
    excluded = frozenset(config.projection_excluded_category_ids)

    code_map: dict[str, str] = {}
    if config.id_strategy == "code":
        if corpus_categories is None:
            raise ValueError(
                f'id_strategy "code" requires corpus categories for {dataset!r}'
            )
        code_map = code_leaf_map(corpus_categories)

    records = load_processed_records(processed_file)
    input_path = Path(processed_file)

    resolver = ClassificationTargetResolver(config, code_leaf_map=code_map)
    status_counts: Counter[str] = Counter()
    missing_leaf_by_name: Counter[str] = Counter()
    path_mismatch_by_id: Counter[str] = Counter()
    code_unresolved_by_leaf: Counter[str] = Counter()
    unresolved: list[dict[str, Any]] = []
    output_records: list[dict[str, Any]] = []

    for record in records:
        result = resolve_record(record, resolver, registry, registry_names, excluded)
        status_counts[result.status.value] += 1
        canonical = _enrich_record(record, dataset, config, result)
        if result.status is ResolutionStatus.RESOLVED:
            assert result.target is not None
            assert result.target.category_id in registry.ids
        else:
            record_id = (
                str(record.get("id", "")) if isinstance(record, Mapping) else ""
            )
            unresolved.append(
                {
                    "id": record_id,
                    "status": result.status.value,
                    "leaf": result.leaf or None,
                    "category_id": result.category_id or None,
                    "reason": result.reason,
                }
            )
            if result.status is ResolutionStatus.MISSING_LEAF:
                missing_leaf_by_name[result.leaf] += 1
            elif result.status is ResolutionStatus.PATH_MISMATCH:
                path_mismatch_by_id[result.category_id] += 1
            elif result.status is ResolutionStatus.CODE_UNRESOLVED:
                code_unresolved_by_leaf[result.leaf] += 1
        output_records.append(canonical)

    unresolved_details: dict[str, Any] = {
        "missing_leaf": {
            "count": sum(missing_leaf_by_name.values()),
            "by_leaf": dict(sorted(missing_leaf_by_name.items())),
        },
        "path_mismatch": {
            "count": sum(path_mismatch_by_id.values()),
            "by_id": dict(sorted(path_mismatch_by_id.items())),
        },
        "code_unresolved": {
            "count": sum(code_unresolved_by_leaf.values()),
            "by_leaf": dict(sorted(code_unresolved_by_leaf.items())),
        },
        "record_ids": [item["id"] for item in unresolved],
    }
    result_summary = CanonicalBuildResult(
        dataset=dataset,
        schema_version=SCHEMA_VERSION,
        input_records=len(records),
        output_records=len(output_records),
        status_counts=dict(sorted(status_counts.items())),
        unresolved_details=unresolved_details,
        # every RESOLVED row was verified against the registry above;
        # anything else stayed unresolved by explicit status
        resolved_targets_in_registry=True,
        registry_size=len(registry.categories),
        registry_file=_as_posix(registry_file),
        registry_source=config.registry_source,
        registry_derivation=config.registry_derivation,
        input_file=_as_posix(input_path),
        output_file=_as_posix(output_file),
        input_sha256=sha256_file(input_path),
    )
    return result_summary, output_records


def write_canonical_dataset(
    result: CanonicalBuildResult,
    output_records: list[dict[str, Any]],
    *,
    overwrite: bool = False,
) -> None:
    """Write canonical all.json + resolution_report.json atomically.

    Must only be called after every selected dataset prepared successfully
    (cross-dataset fail-fast).
    """
    out_all = Path(result.output_file)
    if (out_all.exists() or out_all.with_name("resolution_report.json").exists()) and not overwrite:
        raise FileExistsError(
            f"canonical output exists for {result.dataset}: {out_all.parent} "
            "(pass overwrite=True / --overwrite)"
        )
    out_all.parent.mkdir(parents=True, exist_ok=True)
    report_file = out_all.with_name("resolution_report.json")
    temporary_all = out_all.with_name(f".{out_all.name}.tmp")
    temporary_report = report_file.with_name(f".{report_file.name}.tmp")
    try:
        with temporary_all.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(output_records, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        with temporary_report.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(result.to_mapping(), handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        temporary_all.replace(out_all)
        temporary_report.replace(report_file)
    except BaseException:
        temporary_all.unlink(missing_ok=True)
        temporary_report.unlink(missing_ok=True)
        raise


__all__ = [
    "SCHEMA_VERSION",
    "CanonicalBuildResult",
    "load_corpus_categories_file",
    "load_processed_records",
    "prepare_canonical_dataset",
    "resolve_record",
    "write_canonical_dataset",
]
