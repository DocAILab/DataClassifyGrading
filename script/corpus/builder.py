"""Build retrieval corpus JSON from short classification guide documents."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

from script.preprocessing.processor import normalize_label


DEFAULT_LABEL_COLUMN = "四级子类"
DEFAULT_CONTENT_COLUMN = "内容"


def _load_pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(
            "Corpus construction requires pandas. XLSX input also requires openpyxl."
        ) from exc
    return pd


def _clean_text(value: Any) -> str:
    pd = _load_pandas()
    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    return re.sub(r"\s+", " ", str(value).strip())


def _read_table(path: str | Path):
    pd = _load_pandas()
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Corpus source not found: {source}")

    if source.suffix.lower() == ".xlsx":
        frame = pd.read_excel(source)
    elif source.suffix.lower() == ".csv":
        frame = pd.read_csv(source)
    else:
        raise ValueError("Corpus source must be an .xlsx or .csv file")

    frame.columns = [str(column).strip() for column in frame.columns]
    if frame.columns.duplicated().any():
        duplicates = sorted(set(frame.columns[frame.columns.duplicated()]))
        raise ValueError("Duplicate source columns: " + ", ".join(duplicates))
    return source, frame


def _load_processed_labels(path: str | Path | None) -> tuple[Path | None, set[str]]:
    if path is None:
        return None, set()
    processed = Path(path).expanduser().resolve()
    if not processed.is_file():
        raise FileNotFoundError(f"Processed dataset not found: {processed}")
    try:
        with processed.open("r", encoding="utf-8-sig") as file:
            data = json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid processed JSON: {processed}: {exc}") from exc
    if not isinstance(data, list):
        raise ValueError("Processed JSON must contain a list")

    labels = set()
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Processed item {index} must be an object")
        classification = item.get("classification", {})
        if not isinstance(classification, dict):
            raise ValueError(f"Processed item {index} classification must be an object")
        label = _clean_text(classification.get("level_4"))
        if label:
            labels.add(label)
    return processed, labels


def _display_path(path: Path, project_root: Path | None) -> str:
    if project_root is not None:
        try:
            return path.relative_to(project_root).as_posix()
        except ValueError:
            pass
    return path.name


def _document_id(dataset: str, label: str, content: str) -> str:
    digest = hashlib.sha256(
        f"{dataset}\x1f{label}\x1f{content}".encode("utf-8")
    ).hexdigest()[:24]
    return f"{dataset}-{digest}"


def _write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")


def _write_bundle(
    output_dir: Path,
    corpus: list[dict[str, Any]],
    report: dict[str, Any],
    *,
    overwrite: bool,
) -> None:
    output_dir = output_dir.expanduser().resolve()
    outputs = (output_dir / "corpus.json", output_dir / "build_report.json")
    existing = [path for path in outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing output: "
            + ", ".join(str(path) for path in existing)
            + ". Pass --overwrite to replace it."
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}.staging.", dir=output_dir.parent
    ) as staging_name:
        staging = Path(staging_name)
        staged = (staging / "corpus.json", staging / "build_report.json")
        _write_json(staged[0], corpus)
        _write_json(staged[1], report)

        output_dir.mkdir(parents=True, exist_ok=True)
        for source, destination in zip(staged, outputs):
            os.replace(source, destination)


def build_corpus(
    dataset: str,
    input_file: str | Path,
    output_dir: str | Path,
    *,
    processed_file: str | Path | None = None,
    label_column: str = DEFAULT_LABEL_COLUMN,
    content_column: str = DEFAULT_CONTENT_COLUMN,
    missing_policy: str = "skip",
    overwrite: bool = False,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Build corpus.json and build_report.json from a guide table."""
    if missing_policy not in {"error", "skip"}:
        raise ValueError("missing_policy must be 'error' or 'skip'")

    source, frame = _read_table(input_file)
    label_column = label_column.strip()
    content_column = content_column.strip()
    missing_columns = [
        column for column in (label_column, content_column)
        if column not in frame.columns
    ]
    if missing_columns:
        raise ValueError(
            "Corpus source is missing column(s): " + ", ".join(missing_columns)
        )

    root = Path(project_root).expanduser().resolve() if project_root else None
    processed, processed_labels = _load_processed_labels(processed_file)
    documents: list[dict[str, Any]] = []
    seen_pairs: dict[tuple[str, str], int] = {}
    skipped_missing: list[dict[str, Any]] = []
    duplicate_rows: list[dict[str, Any]] = []
    normalized_label_rows: list[int] = []

    for index, row in frame.iterrows():
        source_row = int(index) + 2
        raw_label = _clean_text(row[label_column])
        content = _clean_text(row[content_column])
        if not raw_label or not content:
            missing = []
            if not raw_label:
                missing.append(label_column)
            if not content:
                missing.append(content_column)
            detail = {"source_row": source_row, "missing_columns": missing}
            if missing_policy == "error":
                raise ValueError(
                    f"Missing {', '.join(missing)} at source row {source_row}"
                )
            skipped_missing.append(detail)
            continue

        label = normalize_label(raw_label)
        if label != raw_label:
            normalized_label_rows.append(source_row)
        pair = (label, content)
        if pair in seen_pairs:
            duplicate_rows.append({
                "source_row": source_row,
                "duplicate_of_source_row": seen_pairs[pair],
            })
            continue
        seen_pairs[pair] = source_row

        metadata: dict[str, Any] = {
            "dataset": dataset,
            "level_4": label,
            "source": _display_path(source, root),
            "source_row": source_row,
        }
        if label != raw_label:
            metadata["source_level_4"] = raw_label
        documents.append({
            "id": _document_id(dataset, label, content),
            "text": content,
            "metadata": metadata,
        })

    if skipped_missing:
        warnings.warn(
            f"Skipping {len(skipped_missing)} corpus row(s) with missing label/content",
            UserWarning,
            stacklevel=2,
        )
    if not documents:
        raise ValueError("No valid corpus documents were generated")

    ids = [document["id"] for document in documents]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Corpus document ID collision detected")

    corpus_labels = {document["metadata"]["level_4"] for document in documents}
    label_counts = Counter(
        document["metadata"]["level_4"] for document in documents
    )
    content_lengths = [len(document["text"]) for document in documents]
    report: dict[str, Any] = {
        "dataset": dataset,
        "source": _display_path(source, root),
        "processed_source": (
            _display_path(processed, root) if processed is not None else None
        ),
        "input_rows": len(frame),
        "exported_documents": len(documents),
        "skipped_missing": {
            "count": len(skipped_missing),
            "rows": skipped_missing,
        },
        "skipped_exact_duplicates": {
            "count": len(duplicate_rows),
            "rows": duplicate_rows,
        },
        "normalized_labels": {
            "count": len(normalized_label_rows),
            "source_rows": normalized_label_rows,
        },
        "corpus_statistics": {
            "unique_level_4": len(corpus_labels),
            "labels_with_multiple_documents": sum(
                count > 1 for count in label_counts.values()
            ),
            "content_length": {
                "min": min(content_lengths),
                "max": max(content_lengths),
                "average": round(sum(content_lengths) / len(content_lengths), 2),
            },
        },
        "processed_label_coverage": {
            "available": processed is not None,
            "processed_unique_level_4": len(processed_labels),
            "overlap": len(corpus_labels & processed_labels),
            "processed_labels_without_corpus": sorted(
                processed_labels - corpus_labels
            ),
            "corpus_labels_unused_by_processed": sorted(
                corpus_labels - processed_labels
            ),
        },
    }

    _write_bundle(Path(output_dir), documents, report, overwrite=overwrite)
    return report
