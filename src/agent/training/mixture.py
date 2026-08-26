"""Deterministic materialization for the formal shougang VeRL release.

The formal release has one dataset and therefore uses a passthrough policy:
source groups are never replicated or downsampled. SFT Stage1/Stage2 rows
remain paired, while cascade RL projects each pair to one Stage1 episode row.
The historical ``build_sqrt_mixture`` name is retained for callers that use
this module as the static-mixture seam.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
from typing import Any, Mapping, Sequence

from agent.release_policy import (
    FORMAL_DATASETS,
    FORMAL_DATASET_SET,
    FORMAL_SAMPLING_POLICY,
)
from agent.training.rl.sample import NATIVE_TOOL_TRAJECTORY_FORMAT


def _hash(*parts: str) -> str:
    return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class MixtureResult:
    rows: tuple[dict[str, Any], ...]
    source_counts: dict[str, int]
    achieved_weights: dict[str, float]
    input_source_counts: dict[str, int]
    family: str
    split: str
    sampling_policy: str = FORMAL_SAMPLING_POLICY


def _source_id(row: Mapping[str, Any], family: str) -> str:
    if family == "sft":
        value = row.get("source_id")
    else:
        extra = row.get("extra_info")
        value = extra.get("source_id") if isinstance(extra, Mapping) else None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{family} row has no source_id")
    return value.strip()


def _stage(row: Mapping[str, Any], family: str) -> str:
    if family == "sft":
        value = row.get("stage")
    else:
        extra = row.get("extra_info")
        value = extra.get("stage") if isinstance(extra, Mapping) else None
    if value not in {"stage1", "stage2"}:
        raise ValueError(f"{family} row has invalid stage")
    return str(value)


def _group_rows(
    rows: Sequence[Mapping[str, Any]], family: str
) -> dict[str, tuple[Mapping[str, Any], ...]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError(f"{family} rows must be mappings")
        grouped.setdefault(_source_id(row, family), []).append(row)
    if not grouped:
        raise ValueError(f"{family} input must contain at least one source")
    result: dict[str, tuple[Mapping[str, Any], ...]] = {}
    for source_id, source_rows in grouped.items():
        by_stage: dict[str, list[Mapping[str, Any]]] = {"stage1": [], "stage2": []}
        for row in source_rows:
            by_stage[_stage(row, family)].append(row)
        if any(len(by_stage[stage]) != 1 for stage in ("stage1", "stage2")):
            raise ValueError(
                f"source {source_id!r} must contain exactly one stage1 and stage2 row"
            )
        result[source_id] = (by_stage["stage1"][0], by_stage["stage2"][0])
    return result


def _targets(counts: Mapping[str, int], split: str) -> dict[str, int]:
    """Return passthrough source counts for every split.

    ``split`` remains an argument for API compatibility with the former
    sqrt-weighted implementation.  With one formal dataset, train/val/test
    all have exactly the input source groups and no materialization replicas.
    """

    del split
    return dict(counts)


def build_sqrt_mixture(
    rows_by_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    family: str,
    split: str,
) -> MixtureResult:
    """Build deterministic rows for ``sft`` or formal ``rl-cascade``."""

    if set(rows_by_dataset) != FORMAL_DATASET_SET:
        raise ValueError(
            "mixture inputs must be exactly the formal dataset set: "
            + ", ".join(FORMAL_DATASETS)
        )
    if family not in {"sft", "rl-cascade"}:
        raise ValueError("mixture family must be sft or rl-cascade")
    if split not in {"train", "val", "test"}:
        raise ValueError("mixture split must be train, val, or test")
    grouped = {
        dataset: _group_rows(tuple(rows_by_dataset[dataset]), family)
        for dataset in FORMAL_DATASETS
    }
    input_counts = {dataset: len(grouped[dataset]) for dataset in FORMAL_DATASETS}
    target_counts = _targets(input_counts, split)
    output: list[dict[str, Any]] = []
    # Hash truncation keeps parquet ids readable, but collisions must never be
    # silently accepted.  SFT intentionally emits one id for its paired rows;
    # formal RL emits exactly one row per episode id.
    seen_mixture_ids: set[str] = set()
    for dataset in FORMAL_DATASETS:
        sources = sorted(
            grouped[dataset], key=lambda source: (_hash(dataset, source), source)
        )
        for replica_index in range(target_counts[dataset]):
            source_id = sources[replica_index % len(sources)]
            cycle = replica_index // len(sources)
            mixture_id = f"mix:{dataset}:{_hash(source_id)[:20]}:{cycle:04d}"
            if mixture_id in seen_mixture_ids:
                raise ValueError("mixture source-id collision detected")
            seen_mixture_ids.add(mixture_id)
            source_rows = grouped[dataset][source_id]
            selected = source_rows if family == "sft" else (source_rows[0],)
            for original in selected:
                row = deepcopy(dict(original))
                if family == "sft":
                    row["source_id"] = mixture_id
                    row["mixture_provenance"] = {
                        "dataset": dataset,
                        "original_source_sha256": _hash(source_id),
                        "replica": cycle,
                    }
                else:
                    extra = deepcopy(dict(row["extra_info"]))
                    if extra.get("trajectory_format") != NATIVE_TOOL_TRAJECTORY_FORMAT:
                        raise ValueError(
                            "formal RL input must use qwen3.5-native-tools-v1"
                        )
                    extra["source_id"] = mixture_id
                    extra["stage"] = "stage1"
                    extra.pop("candidates", None)
                    row["extra_info"] = extra
                    row["data_source"] = f"{dataset}/stage1"
                output.append(row)
    output.sort(
        key=lambda row: _hash(
            split,
            _source_id(row, family),
            _stage(row, family),
        )
    )
    total = sum(target_counts.values())
    achieved = {
        dataset: target_counts[dataset] / total for dataset in FORMAL_DATASETS
    }
    return MixtureResult(
        rows=tuple(output),
        source_counts=target_counts,
        achieved_weights=achieved,
        input_source_counts=input_counts,
        family=family,
        split=split,
        sampling_policy=FORMAL_SAMPLING_POLICY,
    )


__all__ = ["MixtureResult", "build_sqrt_mixture"]
