"""Import the official DCG ``shougang-5415`` split into runtime assets.

The downloaded source IDs are retained as provenance and replaced at the
canonical boundary by this repository's stable record identity.  Category
descriptions are deliberately limited to text derived from the supplied
taxonomy path; they are not presented as an external classification standard.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from agent.hashing import sha256_file
from agent.task.identity import qualified_category_id, stable_record_id


DATASET = "shougang"
SPLITS = ("train", "val", "test")
PATH_FIELDS = ("level_1", "level_2", "level_3", "level_4")
GRADING_LEVELS = ("L1", "L2", "L3", "L4")
EXPECTED_SPLIT_COUNTS = {"train": 4332, "val": 541, "test": 542}
EXPECTED_CATEGORY_COUNT = 193
EXPECTED_FULL_PATH_CATEGORY_COUNT = 203

_METADATA_ALIASES = {
    "database_description": ("database_description", "database_desc"),
    "table_description": ("table_description", "table_desc"),
    "field_description": ("field_description", "field_desc"),
    "field_type": ("field_type", "type"),
}
_IDENTITY_FIELDS = ("database_name", "table_name", "field_name")


@dataclass(frozen=True)
class ShougangPreparedAssets:
    canonical_records: tuple[dict[str, Any], ...]
    registry: dict[str, Any]
    corpus: dict[str, Any]
    dataset_config: dict[str, Any]
    task_config: dict[str, Any]
    grading_config: dict[str, Any]
    report: dict[str, Any]

    def to_mapping(self) -> dict[str, Any]:
        """Return a JSON-compatible defensive copy of all prepared assets."""
        return copy.deepcopy(
            {
                "canonical_records": list(self.canonical_records),
                "registry": self.registry,
                "corpus": self.corpus,
                "dataset_config": self.dataset_config,
                "task_config": self.task_config,
                "grading_config": self.grading_config,
                "report": self.report,
            }
        )


def _load_split(source_dir: Path, split: str) -> tuple[Path, list[Any]]:
    path = source_dir / f"{split}.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing official split file: {path}")
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, list):
        raise ValueError(f"{path} must contain a JSON array")
    return path, value


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _normalized_metadata(raw: Any, *, split: str, index: int) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{split}[{index}] metadata must be an object")
    metadata = copy.deepcopy(dict(raw))
    for field in _IDENTITY_FIELDS:
        value = _text(metadata.get(field))
        if not value:
            raise ValueError(f"{split}[{index}] metadata.{field} must be non-empty")
        metadata[field] = value
    for canonical, aliases in _METADATA_ALIASES.items():
        metadata[canonical] = next(
            (_text(metadata[name]) for name in aliases if _text(metadata.get(name))),
            "",
        )
    return metadata


def _taxonomy_path(raw: Any, *, split: str, index: int) -> tuple[str, ...]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{split}[{index}] classification must be an object")
    path = tuple(_text(raw.get(field)) for field in PATH_FIELDS)
    if not all(path):
        raise ValueError(
            f"{split}[{index}] requires a complete taxonomy path "
            f"({', '.join(PATH_FIELDS)})"
        )
    return path


def _normalize_identity_fields(
    identity_fields: Sequence[str] | None,
) -> tuple[str, ...]:
    fields = tuple(identity_fields or ("level_4",))
    if not fields:
        raise ValueError("identity_fields must not be empty")
    if len(set(fields)) != len(fields):
        raise ValueError("identity_fields must be unique")
    unknown = set(fields) - set(PATH_FIELDS)
    if unknown:
        raise ValueError(
            "identity_fields must be drawn from path fields: "
            + ", ".join(sorted(unknown))
        )
    if "level_4" not in fields:
        raise ValueError("identity_fields must include level_4")
    return fields


def _canonical_record(
    raw: Any,
    *,
    split: str,
    index: int,
    seen_source_ids: set[str],
    seen_record_ids: set[str],
    identity_fields: tuple[str, ...],
) -> tuple[dict[str, Any], tuple[str, ...], str]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{split}[{index}] must be an object")
    source_id = _text(raw.get("id"))
    if not source_id:
        raise ValueError(f"{split}[{index}] source id must be non-empty")
    if source_id in seen_source_ids:
        raise ValueError(f"duplicate source id across splits: {source_id}")
    seen_source_ids.add(source_id)

    metadata = _normalized_metadata(raw.get("metadata"), split=split, index=index)
    record_id = stable_record_id(DATASET, metadata)
    if record_id in seen_record_ids:
        raise ValueError(
            f"duplicate canonical record identity at {split}[{index}]: {record_id}"
        )
    seen_record_ids.add(record_id)

    path = _taxonomy_path(raw.get("classification"), split=split, index=index)
    identity_path = tuple(path[PATH_FIELDS.index(field)] for field in identity_fields)
    category_id = qualified_category_id(DATASET, identity_path)
    data_level = _text(raw.get("data_level"))
    if data_level not in GRADING_LEVELS:
        raise ValueError(
            f"{split}[{index}] data_level must be one of {list(GRADING_LEVELS)}, "
            f"got {data_level!r}"
        )

    record = copy.deepcopy(dict(raw))
    record.update(
        {
            "id": record_id,
            "source_id": source_id,
            "metadata": metadata,
            "schema_version": 2,
            "dataset": DATASET,
            "path_mask": [True, True, True, True],
            "leaf": {"field": "level_4", "value": path[-1]},
            "resolution": {
                "status": "resolved",
                "category_id": category_id,
                "reason": "",
            },
            "resolution_status": "resolved",
            "target": {
                "leaf_level": "level_4",
                "leaf_name": path[-1],
                "category_id": category_id,
                "category_path": list(path),
            },
            "split": split,
            "split_exclusion_reason": None,
            "data_level": data_level,
        }
    )
    return record, path, data_level


def prepare_shougang_5415(
    source_dir: str | Path,
    *,
    strict_counts: bool = True,
    identity_fields: Sequence[str] | None = None,
) -> ShougangPreparedAssets:
    """Validate and prepare the official split without writing any output.

    By default the published 193-class ``level_4`` identity is retained.  The
    optional full ``PATH_FIELDS`` identity produces the corrected 203-class
    taxonomy-path experiment while preserving the exact official splits.
    """
    source = Path(source_dir)
    normalized_identity_fields = _normalize_identity_fields(identity_fields)
    loaded: dict[str, list[Any]] = {}
    input_files: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        path, rows = _load_split(source, split)
        loaded[split] = rows
        input_files[split] = {
            "file": path.name,
            "sha256": sha256_file(path),
            "records": len(rows),
        }

    split_counts = {split: len(loaded[split]) for split in SPLITS}
    if strict_counts and split_counts != EXPECTED_SPLIT_COUNTS:
        raise ValueError(
            f"official split counts must be {EXPECTED_SPLIT_COUNTS}, got {split_counts}"
        )

    seen_source_ids: set[str] = set()
    seen_record_ids: set[str] = set()
    records: list[dict[str, Any]] = []
    paths_by_id: dict[str, set[tuple[str, ...]]] = {}
    categories_by_split: dict[str, set[str]] = {split: set() for split in SPLITS}
    levels_by_split: dict[str, set[str]] = {split: set() for split in SPLITS}
    for split in SPLITS:
        for index, raw in enumerate(loaded[split]):
            record, path, level = _canonical_record(
                raw,
                split=split,
                index=index,
                seen_source_ids=seen_source_ids,
                seen_record_ids=seen_record_ids,
                identity_fields=normalized_identity_fields,
            )
            category_id = record["target"]["category_id"]
            paths_by_id.setdefault(category_id, set()).add(path)
            categories_by_split[split].add(category_id)
            levels_by_split[split].add(level)
            records.append(record)

    category_count = len(paths_by_id)
    expected_category_count = {
        ("level_4",): EXPECTED_CATEGORY_COUNT,
        PATH_FIELDS: EXPECTED_FULL_PATH_CATEGORY_COUNT,
    }.get(normalized_identity_fields)
    if strict_counts and expected_category_count is None:
        raise ValueError(
            "strict_counts supports only level_4 or full taxonomy-path identity"
        )
    if strict_counts and category_count != expected_category_count:
        raise ValueError(
            f"official category count must be {expected_category_count}, "
            f"got {category_count}"
        )
    evaluation_categories = categories_by_split["val"] | categories_by_split["test"]
    unseen_categories = sorted(evaluation_categories - categories_by_split["train"])
    if unseen_categories:
        raise ValueError(
            "evaluation categories absent from train: " + ", ".join(unseen_categories)
        )
    evaluation_levels = levels_by_split["val"] | levels_by_split["test"]
    unseen_levels = sorted(evaluation_levels - levels_by_split["train"])
    if unseen_levels:
        raise ValueError(
            "evaluation data levels absent from train: " + ", ".join(unseen_levels)
        )

    categories = []
    corpus_categories = []
    for category_id, category_paths in sorted(paths_by_id.items()):
        ordered_paths = sorted(category_paths)
        path = ordered_paths[0]
        category = {
            "category_id": category_id,
            "name": path[-1],
            "path": list(path),
        }
        categories.append(category)
        corpus_categories.append(
            {
                **category,
                "description": "Derived taxonomy path: " + " > ".join(path),
                "descriptions": [
                    "Derived taxonomy path: " + " > ".join(alternate)
                    for alternate in ordered_paths[1:]
                ],
            }
        )

    dataset_config = {
        "datasets": {
            DATASET: {
                "leaf_level": "level_4",
                "id_strategy": "path",
                "path_fields": list(PATH_FIELDS),
                "identity_fields": list(normalized_identity_fields),
                "registry_derivation": "dataset-universe",
            }
        }
    }
    task_config = {
        "task_name": "dcg_shougang_field_classification",
        "metadata_fields": [
            "field_name",
            "table_name",
            "field_description",
            "table_description",
        ],
    }
    grading_config = {"levels": list(GRADING_LEVELS), "gt_field": "data_level"}
    report = {
        "status": "passed",
        "dataset": "shougang-5415",
        "canonical_dataset": DATASET,
        "strict_counts": strict_counts,
        "split_counts": split_counts,
        "total_records": len(records),
        "category_count": category_count,
        "identity_fields": list(normalized_identity_fields),
        "taxonomy_path_count": sum(len(paths) for paths in paths_by_id.values()),
        "multi_path_category_count": sum(
            1 for paths in paths_by_id.values() if len(paths) > 1
        ),
        "category_counts_by_split": {
            split: len(categories_by_split[split]) for split in SPLITS
        },
        "data_levels_by_split": {
            split: sorted(levels_by_split[split]) for split in SPLITS
        },
        "unresolved_records": 0,
        "evaluation_categories_absent_from_train": [],
        "evaluation_levels_absent_from_train": [],
        "input_files": input_files,
        "corpus_provenance": "derived_from_dataset_taxonomy_paths",
    }
    return ShougangPreparedAssets(
        canonical_records=tuple(records),
        registry={"categories": categories},
        corpus={
            "dataset": DATASET,
            "provenance": "derived_from_dataset_taxonomy_paths",
            "categories": corpus_categories,
        },
        dataset_config=dataset_config,
        task_config=task_config,
        grading_config=grading_config,
        report=report,
    )


__all__ = [
    "DATASET",
    "EXPECTED_CATEGORY_COUNT",
    "EXPECTED_FULL_PATH_CATEGORY_COUNT",
    "EXPECTED_SPLIT_COUNTS",
    "GRADING_LEVELS",
    "ShougangPreparedAssets",
    "prepare_shougang_5415",
]
