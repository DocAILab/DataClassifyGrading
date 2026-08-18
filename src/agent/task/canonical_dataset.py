"""Canonical dataset construction: resolve every record against the canonical
LeafRegistry and emit a resolution report (stage 3B).

The LeafRegistry is the final constraint on the category universe: a resolver
may produce a target, but the record only counts as resolved when
``target.category_id in registry.ids``. Records whose leaf is corpus-known but
whose identity path differs from the registry (path_mismatch) or whose leaf is
absent from the universe (missing_leaf) stay unresolved and are never
auto-repaired.

Output design (per dataset):
    data/<dataset>/canonical/all.json
        every input record, unchanged (classification untouched), plus:
          - "resolution_status": one of the ResolutionStatus values
          - "target": canonical target (only for resolved records)
    data/<dataset>/canonical/resolution_report.json
        status counts, unresolved details, registry facts, deterministic
        metadata (input sha256; no timestamps, no machine-local paths).
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from agent.task.contracts import CorpusCategory, LeafRegistry, SampleTarget
from agent.task.dataset_config import BUILTIN_DATASET_CONFIGS, DatasetConfig
from agent.task.identity import code_leaf_map
from agent.task.resolver import (
    ClassificationTargetResolver,
    ResolutionResult,
    ResolutionStatus,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _repo_relative(path: str | Path) -> str:
    try:
        return Path(path).resolve().relative_to(_PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return Path(path).as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_registry(registry_file: str | Path) -> LeafRegistry:
    return LeafRegistry.from_path(registry_file)


def load_corpus_categories(corpus_file: str | Path) -> list[CorpusCategory]:
    """Deserialize a canonical corpus JSON back into CorpusCategory objects."""
    with Path(corpus_file).open(encoding="utf-8") as handle:
        data = json.load(handle)
    categories: list[CorpusCategory] = []
    for item in data.get("categories", []):
        categories.append(
            CorpusCategory(
                category_id=str(item.get("category_id", "")).strip(),
                name=str(item.get("name", "")).strip(),
                description=str(item.get("description", "") or ""),
                descriptions=tuple(
                    str(part) for part in item.get("descriptions", ())
                ),
                path=tuple(str(part) for part in item.get("path", ())),
                code=(
                    str(item["code"]).strip()
                    if item.get("code") is not None
                    else None
                ),
                examples=tuple(str(part) for part in item.get("examples", ())),
            )
        )
    return categories


@dataclass(frozen=True)
class CanonicalDatasetResult:
    """Outcome of building one canonical dataset."""

    dataset: str
    input_records: int
    output_records: int
    status_counts: Mapping[str, int]
    unresolved_details: Mapping[str, Any]
    resolved_targets_in_registry: bool
    registry_size: int
    registry_file: str
    registry_source: str | None
    input_file: str
    output_file: str
    input_sha256: str

    def to_mapping(self) -> dict[str, Any]:
        mapping: dict[str, Any] = {
            "dataset": self.dataset,
            "input_records": self.input_records,
            "output_records": self.output_records,
            "status_counts": dict(self.status_counts),
            "resolved_targets_in_registry": self.resolved_targets_in_registry,
            "registry_size": self.registry_size,
            "registry_file": self.registry_file,
            "registry_source": self.registry_source,
            "input_file": self.input_file,
            "output_file": self.output_file,
            "input_sha256": self.input_sha256,
            "unresolved_details": self.unresolved_details,
        }
        return mapping


def resolve_record(
    record: Any,
    resolver: ClassificationTargetResolver,
    registry: LeafRegistry,
    registry_names: set[str],
) -> ResolutionResult:
    """Resolve one record with the LeafRegistry as the final constraint.

    Downgrades a resolver RESOLVED result to PATH_MISMATCH (leaf known in the
    corpus but identity path differs) or MISSING_LEAF (leaf absent from the
    universe) when the generated category_id is not in the registry. Audit
    fields (leaf / category_id) are populated on every downgrade so
    consumers never need to touch ``target`` on non-RESOLVED results.
    """
    result = resolver.resolve_detailed(record)
    if result.status is not ResolutionStatus.RESOLVED or result.target is None:
        return result
    target = result.target
    if target.category_id in registry.ids:
        return result
    if target.leaf_name in registry_names:
        return ResolutionResult(
            ResolutionStatus.PATH_MISMATCH,
            target=target,
            reason=(
                f"leaf {target.leaf_name!r} exists in the corpus but resolved "
                f"identity {target.category_id!r} is not in the registry"
            ),
            leaf=target.leaf_name,
            category_id=target.category_id,
        )
    return ResolutionResult(
        ResolutionStatus.MISSING_LEAF,
        target=target,
        reason=f"leaf {target.leaf_name!r} is absent from the corpus universe",
        leaf=target.leaf_name,
        category_id=target.category_id,
    )


def prepare_canonical_dataset(
    dataset: str,
    *,
    data_dir: str | Path,
    registry_dir: str | Path,
    corpus_dir: str | Path,
) -> tuple[CanonicalDatasetResult, list[dict[str, Any]]]:
    """Load, validate, resolve and build the report for one dataset.

    Pure computation: writes nothing. The caller decides when to write via
    write_canonical_dataset(), which enables cross-dataset fail-fast (no
    partial outputs when any selected dataset fails).
    """
    data_dir = Path(data_dir)
    input_path = data_dir / dataset / "all.json"
    if not input_path.is_file():
        raise FileNotFoundError(f"input dataset not found: {input_path}")

    config = BUILTIN_DATASET_CONFIGS[dataset]
    effective_registry = config.registry_source or dataset
    registry_file = Path(registry_dir) / f"{effective_registry}.registry.json"
    if not registry_file.is_file():
        raise FileNotFoundError(f"registry not found: {registry_file}")
    registry = load_registry(registry_file)
    registry_names = {category.name for category in registry.categories}

    code_map: dict[str, str] = {}
    registry_source: str | None = None
    corpus_file = Path(corpus_dir) / f"{dataset}.corpus.json"
    if corpus_file.is_file():
        with corpus_file.open(encoding="utf-8") as handle:
            corpus_data = json.load(handle)
        registry_source = corpus_data.get("build_report", {}).get("source")
        if config.id_strategy == "code":
            code_map = code_leaf_map(
                load_corpus_categories(corpus_file)
            )
    elif config.id_strategy == "code":
        raise FileNotFoundError(f"canonical corpus not found: {corpus_file}")

    with input_path.open(encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError(f"{input_path} must be a JSON list")

    resolver = ClassificationTargetResolver(config, code_leaf_map=code_map)
    status_counts: Counter[str] = Counter()
    unresolved: list[dict[str, Any]] = []
    missing_leaf_by_name: Counter[str] = Counter()
    path_mismatch_by_id: Counter[str] = Counter()
    code_unresolved_by_leaf: Counter[str] = Counter()
    output_records: list[dict[str, Any]] = []

    for record in records:
        result = resolve_record(record, resolver, registry, registry_names)
        status_counts[result.status.value] += 1
        if isinstance(record, Mapping):
            canonical = copy.deepcopy(dict(record))
        else:
            # non-object record (e.g. a bare string in the JSON array): keep
            # it auditable without corrupting the canonical schema
            canonical = {"record": copy.deepcopy(record)}
        canonical["resolution_status"] = result.status.value
        if result.status is ResolutionStatus.RESOLVED:
            assert result.target is not None
            assert result.target.category_id in registry.ids
            canonical["target"] = _target_mapping(result.target)
        else:
            record_id = (
                str(record.get("id", ""))
                if isinstance(record, Mapping)
                else ""
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

    out_dir = data_dir / dataset / "canonical"
    out_all = out_dir / "all.json"

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
    result = CanonicalDatasetResult(
        dataset=dataset,
        input_records=len(records),
        output_records=len(output_records),
        status_counts=dict(sorted(status_counts.items())),
        unresolved_details=unresolved_details,
        resolved_targets_in_registry=True,
        registry_size=len(registry.categories),
        registry_file=_repo_relative(registry_file),
        registry_source=registry_source,
        input_file=_repo_relative(input_path),
        output_file=_repo_relative(out_all),
        input_sha256=_sha256(input_path),
    )
    return result, output_records


def write_canonical_dataset(
    result: CanonicalDatasetResult,
    output_records: list[dict[str, Any]],
) -> None:
    """Write the canonical all.json + resolution_report.json for a prepared
    dataset. Must only be called after every selected dataset prepared
    successfully (cross-dataset fail-fast)."""
    out_all = Path(result.output_file)
    if not out_all.is_absolute():
        out_all = _PROJECT_ROOT / out_all
    out_report = out_all.parent / "resolution_report.json"
    out_all.parent.mkdir(parents=True, exist_ok=True)
    with out_all.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(output_records, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    with out_report.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(result.to_mapping(), handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def build_canonical_dataset(
    dataset: str,
    *,
    data_dir: str | Path,
    registry_dir: str | Path,
    corpus_dir: str | Path,
    overwrite: bool = False,
) -> CanonicalDatasetResult:
    """Convenience wrapper: prepare + overwrite/existence check + write."""
    data_dir = Path(data_dir)
    out_all = data_dir / dataset / "canonical" / "all.json"
    out_report = data_dir / dataset / "canonical" / "resolution_report.json"
    if (out_all.exists() or out_report.exists()) and not overwrite:
        raise FileExistsError(
            f"canonical output exists for {dataset}: {out_all.parent} "
            "(pass --overwrite)"
        )
    result, output_records = prepare_canonical_dataset(
        dataset,
        data_dir=data_dir,
        registry_dir=registry_dir,
        corpus_dir=corpus_dir,
    )
    write_canonical_dataset(result, output_records)
    return result


def _target_mapping(target: SampleTarget) -> dict[str, Any]:
    return {
        "leaf_level": target.leaf_level,
        "leaf_name": target.leaf_name,
        "category_id": target.category_id,
        "category_path": list(target.category_path),
    }


__all__ = [
    "load_registry",
    "load_corpus_categories",
    "resolve_record",
    "prepare_canonical_dataset",
    "write_canonical_dataset",
    "build_canonical_dataset",
    "CanonicalDatasetResult",
]
