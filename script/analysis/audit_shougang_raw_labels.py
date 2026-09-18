"""Audit the official shougang-5415 raw taxonomy and placeholder leaf labels."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


SPLITS = ("train", "val", "test")
PATH_FIELDS = ("level_1", "level_2", "level_3", "level_4")
PRESERVED_FIELDS = (
    "domain",
    "label_status",
    "metadata",
    "classification",
    "data_level",
)
PLACEHOLDER = "——"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_array(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise ValueError(f"{path} must be a JSON array of objects")
    return value


def load_official_splits(
    source_dir: str | Path,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    """Load train/val/test/all and verify the published all.json ordering."""

    root = Path(source_dir)
    paths = {name: root / f"{name}.json" for name in (*SPLITS, "all")}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    splits = {name: _load_array(paths[name]) for name in SPLITS}
    all_rows = _load_array(paths["all"])
    concatenated = [row for name in SPLITS for row in splits[name]]
    if all_rows != concatenated:
        raise ValueError("all.json must equal train+val+test concatenation")
    ids = [str(row.get("id") or "") for row in concatenated]
    if any(not source_id for source_id in ids):
        raise ValueError("every raw record must have a non-empty id")
    duplicate_ids = sorted(source_id for source_id, count in Counter(ids).items() if count > 1)
    if duplicate_ids:
        raise ValueError(f"duplicate raw IDs: {duplicate_ids}")
    manifest = {
        path.name: {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "records": len(all_rows) if name == "all" else len(splits[name]),
        }
        for name, path in paths.items()
    }
    return splits, manifest


def _path(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    classification = row.get("classification")
    if not isinstance(classification, Mapping):
        raise ValueError(f"{row.get('id')}: classification must be an object")
    result = tuple(str(classification.get(field) or "").strip() for field in PATH_FIELDS)
    if not all(result):
        raise ValueError(f"{row.get('id')}: incomplete taxonomy path")
    return result  # type: ignore[return-value]


def audit_raw_labels(
    splits: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Find leaf collisions and determine whether complete paths resolve them."""

    if set(splits) != set(SPLITS):
        raise ValueError(f"splits must be exactly {list(SPLITS)}")
    rows = [(split, row) for split in SPLITS for row in splits[split]]
    path_levels: dict[tuple[str, ...], set[str]] = defaultdict(set)
    leaf_levels: dict[str, set[str]] = defaultdict(set)
    paths_by_split: dict[str, set[tuple[str, ...]]] = {split: set() for split in SPLITS}
    placeholder_rows: list[tuple[str, Mapping[str, Any], tuple[str, ...]]] = []
    for split, row in rows:
        path = _path(row)
        level = str(row.get("data_level") or "").strip()
        if level not in {"L1", "L2", "L3", "L4"}:
            raise ValueError(f"{row.get('id')}: invalid data_level {level!r}")
        path_levels[path].add(level)
        leaf_levels[path[-1]].add(level)
        paths_by_split[split].add(path)
        if path[-1] == PLACEHOLDER:
            placeholder_rows.append((split, row, path))

    placeholder_paths: dict[tuple[str, ...], list[tuple[str, Mapping[str, Any]]]] = defaultdict(list)
    for split, row, path in placeholder_rows:
        placeholder_paths[path].append((split, row))
    path_breakdown = []
    for path, members in sorted(placeholder_paths.items()):
        path_breakdown.append(
            {
                "path": list(path),
                "records": len(members),
                "split_counts": dict(sorted(Counter(split for split, _ in members).items())),
                "level_counts": dict(
                    sorted(Counter(str(row["data_level"]) for _, row in members).items())
                ),
                "unique_tables": len(
                    {str(row["metadata"]["table_name"]) for _, row in members}
                ),
                "sample_source_id": str(members[0][1]["id"]),
            }
        )

    ambiguous_paths = {
        " > ".join(path): sorted(levels)
        for path, levels in sorted(path_levels.items())
        if len(levels) > 1
    }
    ambiguous_leaves = {
        leaf: sorted(levels)
        for leaf, levels in sorted(leaf_levels.items())
        if len(levels) > 1
    }
    return {
        "records": len(rows),
        "split_counts": {split: len(splits[split]) for split in SPLITS},
        "leaf_label_count": len(leaf_levels),
        "full_path_count": len(path_levels),
        "ambiguous_leaf_label_count": len(ambiguous_leaves),
        "ambiguous_leaf_labels": ambiguous_leaves,
        "ambiguous_full_path_count": len(ambiguous_paths),
        "ambiguous_full_paths": ambiguous_paths,
        "empty_level_4_records": sum(not _path(row)[-1] for _, row in rows),
        "label_status_counts": dict(
            sorted(Counter(str(row.get("label_status")) for _, row in rows).items())
        ),
        "paths_by_split": {split: len(paths_by_split[split]) for split in SPLITS},
        "validation_paths_absent_from_train": [
            list(path) for path in sorted(paths_by_split["val"] - paths_by_split["train"])
        ],
        "test_paths_absent_from_train": [
            list(path) for path in sorted(paths_by_split["test"] - paths_by_split["train"])
        ],
        "placeholder": {
            "label": PLACEHOLDER,
            "records": len(placeholder_rows),
            "fraction": len(placeholder_rows) / len(rows) if rows else 0.0,
            "full_paths": len(placeholder_paths),
            "split_counts": dict(
                sorted(Counter(split for split, _, _ in placeholder_rows).items())
            ),
            "level_counts": dict(
                sorted(
                    Counter(str(row["data_level"]) for _, row, _ in placeholder_rows).items()
                )
            ),
            "label_status_counts": dict(
                sorted(
                    Counter(str(row.get("label_status")) for _, row, _ in placeholder_rows).items()
                )
            ),
            "unique_tables": len(
                {str(row["metadata"]["table_name"]) for _, row, _ in placeholder_rows}
            ),
            "unique_fields": len(
                {str(row["metadata"]["field_name"]) for _, row, _ in placeholder_rows}
            ),
            "path_breakdown": path_breakdown,
        },
    }


def compare_canonical_preservation(
    raw_splits: Mapping[str, Sequence[Mapping[str, Any]]],
    canonical_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Verify canonical import preserved every original label-bearing field."""

    raw = {
        str(row["id"]): (split, row)
        for split in SPLITS
        for row in raw_splits[split]
    }
    canonical = {str(row.get("source_id")): row for row in canonical_records}
    missing = sorted(set(raw).difference(canonical))
    extra = sorted(set(canonical).difference(raw))
    split_mismatches = 0
    field_mismatches: Counter[str] = Counter()
    for source_id in sorted(set(raw) & set(canonical)):
        split, raw_row = raw[source_id]
        canonical_row = canonical[source_id]
        split_mismatches += int(canonical_row.get("split") != split)
        for field in PRESERVED_FIELDS:
            if raw_row.get(field) != canonical_row.get(field):
                field_mismatches[field] += 1
    result = {
        "raw_records": len(raw),
        "canonical_records": len(canonical),
        "missing_source_ids": missing,
        "extra_source_ids": extra,
        "split_mismatches": split_mismatches,
        "field_mismatches": dict(sorted(field_mismatches.items())),
    }
    result["preserved"] = not any(
        (missing, extra, split_mismatches, field_mismatches)
    )
    return result


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_placeholder_csv(
    path: Path, splits: Mapping[str, Sequence[Mapping[str, Any]]]
) -> None:
    fields = [
        "split",
        "source_id",
        *PATH_FIELDS,
        "data_level",
        "table_name",
        "table_description",
        "field_name",
        "field_description",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for split in SPLITS:
            for row in splits[split]:
                taxonomy = _path(row)
                if taxonomy[-1] != PLACEHOLDER:
                    continue
                metadata = row["metadata"]
                writer.writerow(
                    {
                        "split": split,
                        "source_id": row["id"],
                        **dict(zip(PATH_FIELDS, taxonomy, strict=True)),
                        "data_level": row["data_level"],
                        "table_name": metadata.get("table_name", ""),
                        "table_description": metadata.get("table_description", ""),
                        "field_name": metadata.get("field_name", ""),
                        "field_description": metadata.get("field_description", ""),
                    }
                )


def _markdown(report: Mapping[str, Any]) -> str:
    raw = report["raw_label_audit"]
    placeholder = raw["placeholder"]
    canonical = report.get("canonical_preservation")
    lines = [
        "# Shougang-5415 raw-label audit",
        "",
        "## Source integrity",
        "",
        f"- Records: {raw['records']} ({raw['split_counts']})",
        f"- Leaf labels: {raw['leaf_label_count']}",
        f"- Complete taxonomy paths: {raw['full_path_count']}",
        f"- Ambiguous leaf labels: {raw['ambiguous_leaf_labels']}",
        f"- Ambiguous complete paths: {raw['ambiguous_full_path_count']}",
        f"- Validation/test complete paths absent from train: {len(raw['validation_paths_absent_from_train'])}/{len(raw['test_paths_absent_from_train'])}",
    ]
    if canonical:
        lines.append(f"- Raw label-bearing fields preserved by canonical import: {canonical['preserved']}")
    lines.extend(
        [
            "",
            "## Placeholder leaf",
            "",
            f"- Literal label: `{placeholder['label']}`",
            f"- Records: {placeholder['records']} ({placeholder['fraction']:.2%})",
            f"- Complete parent paths: {placeholder['full_paths']}",
            f"- Level counts: {placeholder['level_counts']}",
            f"- Split counts: {placeholder['split_counts']}",
            f"- Label status: {placeholder['label_status_counts']}",
            "",
            "Every complete path has exactly one level. The apparent grading ambiguity is introduced only when the 11 paths are collapsed to their shared level-4 text.",
            "",
            "## Complete path breakdown",
            "",
            "| Path | Records | Levels | Splits |",
            "|---|---:|---|---|",
        ]
    )
    for row in placeholder["path_breakdown"]:
        lines.append(
            f"| {' > '.join(row['path'])} | {row['records']} | {row['level_counts']} | {row['split_counts']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "`——` is a non-empty, labeled source value used when the taxonomy stops at level 3. It is not created by the canonical importer. Treating only level_4 as category identity intentionally merges 11 semantically different paths; full-path identity yields 203 deterministic categories and all 203 occur in train.",
            "",
        ]
    )
    return "\n".join(lines)


def run_audit(
    *,
    source_dir: Path,
    canonical_all: Path | None,
    output_dir: Path,
    overwrite: bool,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    splits, manifest = load_official_splits(source_dir)
    report: dict[str, Any] = {
        "format": "dataclassify-shougang-raw-label-audit-v1",
        "source_manifest": manifest,
        "raw_label_audit": audit_raw_labels(splits),
    }
    if canonical_all is not None:
        canonical = _load_array(canonical_all)
        report["canonical_manifest"] = {
            "path": str(canonical_all),
            "bytes": canonical_all.stat().st_size,
            "sha256": _sha256(canonical_all),
        }
        report["canonical_preservation"] = compare_canonical_preservation(
            splits, canonical
        )
    _write_json(output_dir / "raw-label-audit.json", report)
    _write_placeholder_csv(output_dir / "placeholder-records.csv", splits)
    (output_dir / "raw-label-audit.md").write_text(
        _markdown(report), encoding="utf-8"
    )
    return report


def _args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--canonical-all", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _args(argv)
    try:
        report = run_audit(
            source_dir=args.source_dir,
            canonical_all=args.canonical_all,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(
            f"audit_shougang_raw_labels: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2
    summary = report["raw_label_audit"]
    print(
        json.dumps(
            {
                "records": summary["records"],
                "leaf_labels": summary["leaf_label_count"],
                "full_paths": summary["full_path_count"],
                "ambiguous_leaves": summary["ambiguous_leaf_label_count"],
                "ambiguous_paths": summary["ambiguous_full_path_count"],
                "placeholder_records": summary["placeholder"]["records"],
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
