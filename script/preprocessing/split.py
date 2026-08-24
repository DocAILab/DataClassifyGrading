"""Deterministic random or group-aware dataset splitting."""

from __future__ import annotations

import json
import os
import random
import tempfile
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SPLIT_ALGORITHM_VERSION = "1"  # bump when assignment logic changes

LABEL_FIELDS = (
    "classification.level_1",
    "classification.level_2",
    "classification.level_3",
    "classification.level_4",
    "data_level",
)
MISSING = object()


def _load_json_list(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Input file not found: {source}")
    try:
        with source.open("r", encoding="utf-8-sig") as file:
            data = json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid input JSON: {source}: {exc}") from exc
    if not isinstance(data, list):
        raise ValueError("Input JSON must contain a list")
    if not data:
        raise ValueError("Input dataset is empty")
    if not all(isinstance(item, dict) for item in data):
        raise ValueError("Every dataset item must be a JSON object")
    return data


def _atomic_write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
            file.write("\n")
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _get_nested(item: dict[str, Any], key: str) -> Any:
    value: Any = item
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            return MISSING
        value = value[part]
    return value


def _label_value(item: dict[str, Any], field: str) -> str:
    value = _get_nested(item, field)
    return "" if value is MISSING or value is None else str(value)


def _missing_group_indexes(
    data: list[dict[str, Any]], group_key: str
) -> list[int]:
    missing = []
    for index, item in enumerate(data):
        value = _get_nested(item, group_key)
        if (
            value is MISSING
            or value is None
            or (isinstance(value, str) and not value.strip())
        ):
            missing.append(index)
    return missing


def _allocate_counts(total: int, ratios: tuple[float, float, float]) -> list[int]:
    raw = [total * ratio for ratio in ratios]
    counts = [int(value) for value in raw]
    remaining = total - sum(counts)
    order = sorted(
        range(3),
        key=lambda index: (raw[index] - counts[index], ratios[index]),
        reverse=True,
    )
    for index in order[:remaining]:
        counts[index] += 1
    if total and ratios[0] and not counts[0]:
        donor = max(
            (index for index in (1, 2) if counts[index]),
            key=counts.__getitem__,
            default=None,
        )
        if donor is not None:
            counts[donor] -= 1
            counts[0] += 1
    return counts


def _split_random(
    data: list[dict[str, Any]],
    ratios: tuple[float, float, float],
    seed: int,
) -> tuple[list[dict[str, Any]], ...]:
    rng = random.Random(seed)
    target_counts = _allocate_counts(len(data), ratios)
    remaining = target_counts.copy()
    strata: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for item in data:
        strata[tuple(_label_value(item, field) for field in LABEL_FIELDS)].append(item)

    splits: list[list[dict[str, Any]]] = [[], [], []]
    stratum_items = list(strata.values())
    rng.shuffle(stratum_items)
    stratum_items.sort(key=len, reverse=True)
    for items in stratum_items:
        items = items.copy()
        rng.shuffle(items)
        expected = [len(items) * ratio for ratio in ratios]
        assigned = [0, 0, 0]
        for item in items:
            candidates = [index for index in range(3) if remaining[index] > 0]
            target_index = min(
                candidates,
                key=lambda index: (
                    (
                        (assigned[index] + 1 - expected[index]) ** 2
                        - (assigned[index] - expected[index]) ** 2
                    )
                    / max(expected[index], 1),
                    -remaining[index] / max(target_counts[index], 1),
                    index,
                ),
            )
            splits[target_index].append(item)
            assigned[target_index] += 1
            remaining[target_index] -= 1
    for split in splits:
        rng.shuffle(split)
    if remaining != [0, 0, 0]:
        raise RuntimeError(f"Internal split allocation error: remaining={remaining}")
    return tuple(splits)


def _split_group(
    data: list[dict[str, Any]],
    group_key: str,
    ratios: tuple[float, float, float],
    seed: int,
) -> tuple[list[dict[str, Any]], ...]:
    groups: dict[Any, list[dict[str, Any]]] = {}
    missing: list[int] = []
    for index, item in enumerate(data):
        key = _get_nested(item, group_key)
        if (
            key is MISSING
            or key is None
            or (isinstance(key, str) and not key.strip())
        ):
            missing.append(index)
            continue
        try:
            hash(key)
        except TypeError as exc:
            raise ValueError(
                f"group key '{group_key}' must resolve to a scalar at item {index}"
            ) from exc
        groups.setdefault(key, []).append(item)
    if missing:
        examples = ", ".join(map(str, missing[:10]))
        raise ValueError(
            f"group key '{group_key}' is missing or empty at item(s): {examples}"
        )

    targets = [len(data) * ratio for ratio in ratios]
    splits: list[list[dict[str, Any]]] = [[], [], []]
    sizes = [0, 0, 0]
    active = [index for index, ratio in enumerate(ratios) if ratio > 0]
    if len(groups) < len(active):
        raise ValueError(
            f"Only {len(groups)} groups are available for {len(active)} non-empty splits"
        )

    total_labels: Counter[tuple[str, str]] = Counter()
    labels_by_group: dict[Any, Counter[tuple[str, str]]] = {}
    for group_name, items in groups.items():
        labels = Counter(
            (field, _label_value(item, field))
            for item in items
            for field in LABEL_FIELDS
        )
        labels_by_group[group_name] = labels
        total_labels.update(labels)
    label_targets = [
        {
            label: count * ratio
            for label, count in total_labels.items()
        }
        for ratio in ratios
    ]
    split_labels: list[Counter[tuple[str, str]]] = [
        Counter(), Counter(), Counter()
    ]

    rng = random.Random(seed)
    group_items = list(groups.items())
    rng.shuffle(group_items)
    group_items.sort(key=lambda pair: len(pair[1]), reverse=True)
    for position, (group_name, items) in enumerate(group_items):
        labels = labels_by_group[group_name]
        if position < len(active):
            target_index = active[position]
        else:
            target_index = min(
                active,
                key=lambda candidate: (
                    sum(
                        (
                            sizes[index]
                            + (len(items) if index == candidate else 0)
                            - targets[index]
                        ) ** 2
                        / max(targets[index], 1)
                        for index in active
                    )
                    + sum(
                        (
                            split_labels[index][label]
                            + (count if index == candidate else 0)
                            - label_targets[index][label]
                        ) ** 2
                        / max(label_targets[index][label], 1)
                        for index in active
                        for label, count in labels.items()
                    )
                    / max(len(labels), 1),
                    sizes[candidate] / max(targets[candidate], 1),
                    candidate,
                ),
            )
        splits[target_index].extend(items)
        sizes[target_index] += len(items)
        split_labels[target_index].update(labels)
    return tuple(splits)


def build_split_report(splits: tuple[list[dict[str, Any]], ...]) -> dict[str, Any]:
    names = ("train", "val", "test")
    report: dict[str, Any] = {
        "sizes": dict(zip(names, map(len, splits))),
        "distributions": {},
        "warnings": [],
    }
    for field in LABEL_FIELDS:
        distributions = {}
        label_sets = []
        for name, split in zip(names, splits):
            counter = Counter(_label_value(item, field) or "<EMPTY>" for item in split)
            distributions[name] = dict(sorted(counter.items()))
            label_sets.append(set(counter) - {"<EMPTY>"})
        report["distributions"][field] = distributions
        for warning_type, labels in (
            ("labels_missing_from_train", (label_sets[1] | label_sets[2]) - label_sets[0]),
            ("train_labels_missing_from_val", label_sets[0] - label_sets[1]),
            ("train_labels_missing_from_test", label_sets[0] - label_sets[2]),
        ):
            if labels:
                report["warnings"].append({
                    "type": warning_type, "field": field, "labels": sorted(labels)
                })
    return report


def stratified_split(
    data: list[dict[str, Any]],
    ratios: tuple[float, float, float],
    seed: int,
) -> tuple[list[dict[str, Any]], ...]:
    """Public wrapper around the stratified random split core."""
    return _split_random(data, ratios, seed)


def group_split(
    data: list[dict[str, Any]],
    group_key: str,
    ratios: tuple[float, float, float],
    seed: int,
) -> tuple[list[dict[str, Any]], ...]:
    """Public wrapper around the group-aware split core.

    Assumes every item carries a non-empty ``group_key``; callers filter
    and report exclusions themselves (see the canonical splitter).
    """
    missing = _missing_group_indexes(data, group_key)
    if missing:
        raise ValueError(
            f"group key '{group_key}' is missing or empty at item(s): "
            f", ".join(map(str, missing[:5]))
        )
    return _split_group(data, group_key, ratios, seed)


def canonical_record_order(data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return records sorted by stable identity (id, then full content).

    Sorting before splitting removes input-row-order sensitivity: the same
    logical dataset always yields the same splits regardless of the row
    order in the source file.
    """
    return sorted(
        data,
        key=lambda item: (
            str(item.get("id", "") or ""),
            json.dumps(item, ensure_ascii=False, sort_keys=True),
        ),
    )


def split_dataset(
    input_file: str | Path,
    output_dir: str | Path,
    *,
    split_type: str = "random",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42,
    group_key: str | None = None,
    missing_group_policy: str = "error",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Split a normalized JSON list and return the generated report."""
    ratios = (train_ratio, val_ratio, test_ratio)
    if any(ratio < 0 or ratio > 1 for ratio in ratios):
        raise ValueError("Every split ratio must be between 0 and 1")
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("train, val, and test ratios must sum to 1")
    if train_ratio == 0:
        raise ValueError("train ratio must be greater than zero")
    if missing_group_policy not in {"error", "skip"}:
        raise ValueError("missing_group_policy must be 'error' or 'skip'")

    data = _load_json_list(input_file)
    input_size = len(data)
    skipped_group_indexes: list[int] = []
    data = canonical_record_order(data)
    if split_type == "random":
        splits = _split_random(data, ratios, random_seed)
    elif split_type == "group":
        if not group_key:
            raise ValueError("group_key is required for a group split")
        skipped_group_indexes = _missing_group_indexes(data, group_key)
        if skipped_group_indexes:
            examples = ", ".join(map(str, skipped_group_indexes[:10]))
            suffix = "..." if len(skipped_group_indexes) > 10 else ""
            message = (
                f"group key '{group_key}' is missing or empty at item(s): "
                f"{examples}{suffix}"
            )
            if missing_group_policy == "error":
                raise ValueError(message)
            warnings.warn(
                f"Skipping {len(skipped_group_indexes)} record(s): {message}",
                UserWarning,
                stacklevel=2,
            )
            skipped = set(skipped_group_indexes)
            data = [item for index, item in enumerate(data) if index not in skipped]
            if not data:
                raise ValueError(
                    f"No records remain after skipping empty group key '{group_key}'"
                )
        splits = _split_group(data, group_key, ratios, random_seed)
    else:
        raise ValueError("split_type must be 'random' or 'group'")

    destination = Path(output_dir).expanduser().resolve()
    output_paths = [destination / name for name in (
        "train.json", "val.json", "test.json", "split_report.json"
    )]
    existing = [path for path in output_paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing output: "
            + ", ".join(str(path) for path in existing)
            + ". Pass --overwrite to replace it."
        )

    report = build_split_report(splits)
    report["input_size"] = input_size
    report["included_size"] = len(data)
    # reproducibility anchors: everything needed to re-run byte-identically
    report["seed"] = random_seed
    report["split_type"] = split_type
    if split_type == "group":
        report["group_key"] = group_key
    report["ratios"] = {
        "train": train_ratio,
        "val": val_ratio,
        "test": test_ratio,
    }
    report["order_rule"] = "id-ascending"
    report["algorithm_version"] = SPLIT_ALGORITHM_VERSION
    report["skipped_records"] = {
        "count": len(skipped_group_indexes),
        "reasons": (
            [{
                "reason": f"missing_group_key:{group_key}",
                "count": len(skipped_group_indexes),
                "item_indexes": skipped_group_indexes,
            }]
            if skipped_group_indexes
            else []
        ),
    }
    # Fully serialize every output before replacing any destination file.
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}.staging.", dir=destination.parent
    ) as staging_name:
        staging = Path(staging_name)
        staged_paths = [staging / path.name for path in output_paths]
        for split, path in zip(splits, staged_paths[:3]):
            _atomic_write_json(split, path)
        _atomic_write_json(report, staged_paths[3])

        destination.mkdir(parents=True, exist_ok=True)
        for staged, output in zip(staged_paths, output_paths):
            os.replace(staged, output)
    return report
