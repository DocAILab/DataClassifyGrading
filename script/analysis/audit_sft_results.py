"""Audit SFT split leakage, frozen artifacts, and two-stage prediction errors."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
from html import escape
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence
import unicodedata


METADATA_FIELDS = (
    "field_name",
    "table_name",
    "field_description",
    "table_description",
)
NEAR_THRESHOLDS = (0.90, 0.95, 0.99)


def normalize_text(value: object) -> str:
    """Return a comparison key insensitive to case, spacing, and punctuation."""

    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    return "".join(character for character in text if character.isalnum())


def _metadata(record: Mapping[str, Any]) -> dict[str, str]:
    value = record.get("metadata")
    if not isinstance(value, Mapping):
        raise ValueError(f"{record.get('source_id')}: metadata must be an object")
    return {field: str(value.get(field) or "") for field in METADATA_FIELDS}


def _record_projection(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source_id": str(record["source_id"]),
        "metadata": _metadata(record),
        "ground_truth": str(record["ground_truth"]),
        "ground_truth_level": str(record["ground_truth_level"]),
    }


def collapse_stage_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Collapse paired Stage-1/Stage-2 rows into one record per source."""

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        source_id = row.get("source_id")
        if not isinstance(source_id, str) or not source_id:
            raise ValueError("every parquet row must have a non-empty source_id")
        grouped[source_id].append(row)

    collapsed: list[dict[str, Any]] = []
    for source_id in sorted(grouped):
        pair = grouped[source_id]
        stages = [row.get("stage") for row in pair]
        if len(pair) != 2 or sorted(stages) != ["stage1", "stage2"]:
            raise ValueError(
                f"{source_id}: expected one stage1 and one stage2 row, got {stages}"
            )
        projections = [_record_projection(row) for row in pair]
        if projections[0] != projections[1]:
            raise ValueError(f"{source_id}: stage rows disagree on labels or metadata")
        collapsed.append(projections[0])
    return collapsed


def load_split(path: str | Path) -> list[dict[str, Any]]:
    """Read a VERL messages parquet and return one record per source."""

    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:  # pragma: no cover - actionable CLI guard
        raise RuntimeError("pyarrow is required to read SFT parquet files") from exc
    table = parquet.read_table(path)
    required = {
        "stage",
        "source_id",
        "metadata",
        "ground_truth",
        "ground_truth_level",
    }
    missing = sorted(required.difference(table.column_names))
    if missing:
        raise ValueError(f"{path}: missing columns: {', '.join(missing)}")
    return collapse_stage_rows(table.select(sorted(required)).to_pylist())


def _raw_metadata_key(record: Mapping[str, Any]) -> tuple[str, ...]:
    metadata = _metadata(record)
    return tuple(metadata[field] for field in METADATA_FIELDS)


def _normalized_metadata_key(record: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(normalize_text(value) for value in _raw_metadata_key(record))


def _document(record: Mapping[str, Any]) -> str:
    return "|".join(_normalized_metadata_key(record))


def _ngrams(text: str, width: int = 3) -> frozenset[str]:
    compact = f"^{text}$"
    if len(compact) <= width:
        return frozenset({compact})
    return frozenset(compact[index : index + width] for index in range(len(compact) - width + 1))


def _ngram_similarity(left: frozenset[str], right: frozenset[str]) -> float:
    """Return the Sorensen-Dice similarity of two character-ngram sets."""

    if not left and not right:
        return 1.0
    total = len(left) + len(right)
    return 2.0 * len(left & right) / total if total else 0.0


def audit_split_overlap(
    train_records: Sequence[Mapping[str, Any]],
    test_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Measure exact, normalized, table-group, and character-ngram overlap."""

    train_ids = {str(record["source_id"]) for record in train_records}
    raw_keys = {_raw_metadata_key(record) for record in train_records}
    normalized_keys = {_normalized_metadata_key(record) for record in train_records}
    table_keys = {
        normalize_text(_metadata(record)["table_name"]) for record in train_records
    }
    field_table_keys = {
        (
            normalize_text(_metadata(record)["field_name"]),
            normalize_text(_metadata(record)["table_name"]),
        )
        for record in train_records
    }
    field_keys = {
        normalize_text(_metadata(record)["field_name"]) for record in train_records
    }
    description_keys = {
        normalize_text(_metadata(record)["field_description"])
        for record in train_records
        if normalize_text(_metadata(record)["field_description"])
    }

    train_ngrams: list[frozenset[str]] = []
    inverted: dict[str, set[int]] = defaultdict(set)
    for index, record in enumerate(train_records):
        grams = _ngrams(_document(record))
        train_ngrams.append(grams)
        for gram in grams:
            inverted[gram].add(index)

    per_test: list[dict[str, Any]] = []
    for record in test_records:
        metadata = _metadata(record)
        grams = _ngrams(_document(record))
        candidate_indices: set[int] = set()
        for gram in grams:
            candidate_indices.update(inverted.get(gram, ()))
        nearest_index: int | None = None
        nearest_similarity = 0.0
        for index in sorted(candidate_indices):
            similarity = _ngram_similarity(grams, train_ngrams[index])
            if similarity > nearest_similarity:
                nearest_index = index
                nearest_similarity = similarity
        nearest = train_records[nearest_index] if nearest_index is not None else None
        field_table = (
            normalize_text(metadata["field_name"]),
            normalize_text(metadata["table_name"]),
        )
        per_test.append(
            {
                "source_id": str(record["source_id"]),
                "source_id_overlap": str(record["source_id"]) in train_ids,
                "exact_metadata_overlap": _raw_metadata_key(record) in raw_keys,
                "normalized_metadata_overlap": _normalized_metadata_key(record)
                in normalized_keys,
                "normalized_table_overlap": field_table[1] in table_keys,
                "normalized_field_table_overlap": field_table in field_table_keys,
                "normalized_field_name_overlap": field_table[0] in field_keys,
                "normalized_field_description_overlap": bool(
                    normalize_text(metadata["field_description"])
                )
                and normalize_text(metadata["field_description"]) in description_keys,
                "nearest_train_source_id": (
                    str(nearest["source_id"]) if nearest is not None else None
                ),
                "nearest_train_similarity": nearest_similarity,
                "nearest_train_same_label": (
                    bool(nearest)
                    and str(nearest["ground_truth"]) == str(record["ground_truth"])
                ),
            }
        )

    summary: dict[str, Any] = {
        "train_sources": len(train_records),
        "test_sources": len(test_records),
    }
    boolean_fields = (
        "source_id_overlap",
        "exact_metadata_overlap",
        "normalized_metadata_overlap",
        "normalized_table_overlap",
        "normalized_field_table_overlap",
        "normalized_field_name_overlap",
        "normalized_field_description_overlap",
    )
    for field in boolean_fields:
        summary[field] = sum(bool(row[field]) for row in per_test)
    for threshold in NEAR_THRESHOLDS:
        suffix = str(threshold).replace(".", "_")
        summary[f"near_duplicate_at_{suffix}"] = sum(
            float(row["nearest_train_similarity"]) >= threshold for row in per_test
        )
    similarities = sorted(float(row["nearest_train_similarity"]) for row in per_test)
    summary["nearest_similarity_mean"] = (
        sum(similarities) / len(similarities) if similarities else 0.0
    )
    summary["nearest_similarity_median"] = (
        similarities[len(similarities) // 2] if similarities else 0.0
    )
    return {"summary": summary, "per_test": per_test}


def performance_slices(
    overlap_rows: Sequence[Mapping[str, Any]],
    report_rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, float | int]]:
    """Compute strict-joint accuracy on leakage-risk and clean complements."""

    predictions = {str(row["source_id"]): bool(row.get("e2e_correct")) for row in report_rows}
    if {str(row["source_id"]) for row in overlap_rows} != set(predictions):
        raise ValueError("overlap rows and prediction rows must have identical source IDs")

    def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
        correct = sum(predictions[str(row["source_id"])] for row in rows)
        return {
            "sources": len(rows),
            "correct": correct,
            "accuracy": correct / len(rows) if rows else 0.0,
        }

    result = {
        "seen_table": summarize(
            [row for row in overlap_rows if bool(row["normalized_table_overlap"])]
        ),
        "unseen_table": summarize(
            [row for row in overlap_rows if not bool(row["normalized_table_overlap"])]
        ),
    }
    for threshold in NEAR_THRESHOLDS:
        suffix = str(threshold).replace(".", "_")
        result[f"similarity_at_least_{suffix}"] = summarize(
            [
                row
                for row in overlap_rows
                if float(row["nearest_train_similarity"]) >= threshold
            ]
        )
        result[f"similarity_below_{suffix}"] = summarize(
            [
                row
                for row in overlap_rows
                if float(row["nearest_train_similarity"]) < threshold
            ]
        )
    return result


def _classification_metrics(
    truths: Sequence[str], predictions: Sequence[str]
) -> tuple[float, list[dict[str, Any]], list[str], list[list[int]]]:
    labels = sorted(set(truths) | set(predictions))
    index = {label: position for position, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for truth, prediction in zip(truths, predictions, strict=True):
        matrix[index[truth]][index[prediction]] += 1
    rows: list[dict[str, Any]] = []
    f1_values: list[float] = []
    for position, label in enumerate(labels):
        true_positive = matrix[position][position]
        support = sum(matrix[position])
        predicted = sum(row[position] for row in matrix)
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / support if support else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        if support:
            f1_values.append(f1)
        rows.append(
            {
                "label": label,
                "support": support,
                "predicted": predicted,
                "correct": true_positive,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return (
        sum(f1_values) / len(f1_values) if f1_values else 0.0,
        rows,
        labels,
        matrix,
    )


def audit_predictions(
    test_records: Sequence[Mapping[str, Any]], report: Mapping[str, Any]
) -> dict[str, Any]:
    """Recompute metrics and assign mutually interpretable error types."""

    report_rows = report.get("per_source")
    if not isinstance(report_rows, list):
        raise ValueError("evaluation report must contain a per_source list")
    by_id = {str(record["source_id"]): record for record in test_records}
    predictions = {str(row.get("source_id")): row for row in report_rows}
    if set(by_id) != set(predictions):
        missing = sorted(set(by_id).difference(predictions))
        extra = sorted(set(predictions).difference(by_id))
        raise ValueError(f"prediction/test source mismatch: missing={missing}, extra={extra}")

    truths: list[str] = []
    predicted: list[str] = []
    level_truths: list[str] = []
    level_predictions: list[str] = []
    errors: list[dict[str, Any]] = []
    recalled_count = 0
    correct_when_recalled = 0
    stage1_miss = 0
    stage2_selection_error = 0
    level_only_error = 0

    ordered_rows = [predictions[str(record["source_id"])] for record in test_records]
    for record, row in zip(test_records, ordered_rows, strict=True):
        truth = str(record["ground_truth"])
        level_truth = str(record["ground_truth_level"])
        if str(row.get("ground_truth")) != truth:
            raise ValueError(f"{record['source_id']}: report ground truth mismatch")
        if str(row.get("ground_truth_level")) != level_truth:
            raise ValueError(f"{record['source_id']}: report level mismatch")
        decision = str(row.get("final_decision") or "<invalid>")
        level = str(row.get("predicted_level") or "<invalid>")
        recalled = bool(row.get("recalled"))
        leaf_correct = decision == truth
        level_correct = level == level_truth
        truths.append(truth)
        predicted.append(decision)
        level_truths.append(level_truth)
        level_predictions.append(level)
        if recalled:
            recalled_count += 1
            correct_when_recalled += int(leaf_correct)

        error_parts: list[str] = []
        if not recalled:
            stage1_miss += 1
            error_parts.append("stage1_miss")
        elif not leaf_correct:
            stage2_selection_error += 1
            error_parts.append("stage2_selection_error")
        if leaf_correct and not level_correct:
            level_only_error += 1
            error_parts.append("level_only_error")
        elif not level_correct:
            error_parts.append("level_error")
        if error_parts:
            errors.append(
                {
                    "source_id": str(record["source_id"]),
                    "error_type": "+".join(error_parts),
                    **_metadata(record),
                    "ground_truth": truth,
                    "prediction": decision,
                    "ground_truth_level": level_truth,
                    "predicted_level": level,
                    "recalled": recalled,
                    "predicted_top5": row.get("predicted_top5", []),
                }
            )

    category_macro_f1, per_class, labels, matrix = _classification_metrics(
        truths, predicted
    )
    level_macro_f1, per_level, level_labels, level_matrix = _classification_metrics(
        level_truths, level_predictions
    )
    per_class_by_label = {row["label"]: row for row in per_class}
    for row, report_row in zip(test_records, ordered_rows, strict=True):
        per_class_by_label[str(row["ground_truth"])].setdefault(
            "stage1_recalled", 0
        )
        per_class_by_label[str(row["ground_truth"])]["stage1_recalled"] += int(
            bool(report_row.get("recalled"))
        )
    for row in per_class:
        row["stage1_recall_at_5"] = (
            row.get("stage1_recalled", 0) / row["support"] if row["support"] else 0.0
        )

    confusion_pairs = Counter(
        (truth, prediction)
        for truth, prediction in zip(truths, predicted, strict=True)
        if truth != prediction
    )
    summary = {
        "sources": len(test_records),
        "stage1_miss": stage1_miss,
        "stage1_recall_at_5": 1.0 - stage1_miss / len(test_records),
        "stage2_selection_error": stage2_selection_error,
        "stage2_oracle_accuracy_when_recalled": (
            correct_when_recalled / recalled_count if recalled_count else 0.0
        ),
        "category_correct": sum(t == p for t, p in zip(truths, predicted, strict=True)),
        "category_accuracy": sum(t == p for t, p in zip(truths, predicted, strict=True))
        / len(test_records),
        "category_macro_f1": category_macro_f1,
        "level_correct": sum(
            t == p for t, p in zip(level_truths, level_predictions, strict=True)
        ),
        "level_accuracy": sum(
            t == p for t, p in zip(level_truths, level_predictions, strict=True)
        )
        / len(test_records),
        "level_macro_f1": level_macro_f1,
        "level_only_error": level_only_error,
        "strict_joint_correct": sum(
            truth == prediction and level_truth == level_prediction
            for truth, prediction, level_truth, level_prediction in zip(
                truths, predicted, level_truths, level_predictions, strict=True
            )
        ),
    }
    summary["strict_joint_accuracy"] = summary["strict_joint_correct"] / len(
        test_records
    )
    return {
        "summary": summary,
        "per_class": per_class,
        "per_level": per_level,
        "category_labels": labels,
        "category_confusion_matrix": matrix,
        "level_labels": level_labels,
        "level_confusion_matrix": level_matrix,
        "top_confusions": [
            {"ground_truth": truth, "prediction": prediction, "count": count}
            for (truth, prediction), count in confusion_pairs.most_common()
        ],
        "errors": errors,
    }


def audit_level_mapping(
    train_records: Sequence[Mapping[str, Any]],
    test_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Evaluate how much grading is explained by a train-majority category map."""

    counts: dict[str, Counter[str]] = defaultdict(Counter)
    global_counts: Counter[str] = Counter()
    for record in train_records:
        category = str(record["ground_truth"])
        level = str(record["ground_truth_level"])
        counts[category][level] += 1
        global_counts[level] += 1
    majority = {
        category: counter.most_common(1)[0][0] for category, counter in counts.items()
    }
    fallback = global_counts.most_common(1)[0][0]
    correct = 0
    unseen = 0
    for record in test_records:
        category = str(record["ground_truth"])
        if category not in majority:
            unseen += 1
        predicted = majority.get(category, fallback)
        correct += int(predicted == str(record["ground_truth_level"]))
    ambiguous = {
        category: dict(counter)
        for category, counter in counts.items()
        if len(counter) > 1
    }
    return {
        "train_categories": len(counts),
        "ambiguous_train_categories": len(ambiguous),
        "ambiguous_category_levels": ambiguous,
        "test_unseen_categories": unseen,
        "train_majority_category_to_level_accuracy": correct / len(test_records),
        "train_global_majority_level": fallback,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = list(rows[0]) if rows else ["empty"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            normalized = {
                key: json.dumps(value, ensure_ascii=False)
                if isinstance(value, (list, dict))
                else value
                for key, value in row.items()
            }
            writer.writerow(normalized)


def _write_matrix(path: Path, labels: Sequence[str], matrix: Sequence[Sequence[int]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ground_truth\\prediction", *labels])
        for label, row in zip(labels, matrix, strict=True):
            writer.writerow([label, *row])


def _short_label(label: str) -> str:
    return label.split(":", 1)[-1]


def _write_figures(output: Path, prediction: Mapping[str, Any]) -> None:
    """Write dependency-free, colorblind-safe SVG confusion heatmaps."""

    def color(value: int, maximum: int) -> str:
        anchors = ("#00204c", "#414487", "#2a788e", "#22a884", "#7ad151", "#fde725")
        if maximum <= 0:
            return anchors[0]
        fraction = max(0.0, min(1.0, value / maximum))
        return anchors[min(int(fraction * (len(anchors) - 1)), len(anchors) - 1)]

    def svg_heatmap(
        path: Path,
        labels: Sequence[str],
        matrix: Sequence[Sequence[int]],
        title: str,
        *,
        cell: int,
        left: int,
        top: int,
    ) -> None:
        maximum = max((max(row) for row in matrix), default=0)
        width = left + cell * len(labels) + 35
        height = top + cell * len(labels) + 55
        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="white"/>',
            '<style>text{font-family:"Microsoft YaHei","Noto Sans CJK SC",Arial,sans-serif}</style>',
            f'<text x="{width / 2:.1f}" y="22" text-anchor="middle" font-size="14" font-weight="bold">{escape(title)}</text>',
            f'<text x="{width / 2:.1f}" y="{height - 8}" text-anchor="middle" font-size="11">Predicted label</text>',
            f'<text x="14" y="{top + cell * len(labels) / 2:.1f}" text-anchor="middle" font-size="11" transform="rotate(-90 14 {top + cell * len(labels) / 2:.1f})">Ground-truth label</text>',
        ]
        for index, label in enumerate(labels):
            short = escape(_short_label(label))
            x = left + index * cell + cell / 2
            y = top - 6
            parts.append(
                f'<text x="{x:.1f}" y="{y}" text-anchor="end" font-size="7" transform="rotate(-60 {x:.1f} {y})">{short}</text>'
            )
            parts.append(
                f'<text x="{left - 7}" y="{top + index * cell + cell * 0.70:.1f}" text-anchor="end" font-size="7">{short}</text>'
            )
        for row_index, row in enumerate(matrix):
            for column_index, value in enumerate(row):
                x = left + column_index * cell
                y = top + row_index * cell
                fill = color(int(value), maximum)
                parts.append(
                    f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{fill}" stroke="white" stroke-width="0.4"/>'
                )
                if len(labels) <= 12:
                    text_color = "white" if maximum and value / maximum < 0.7 else "black"
                    parts.append(
                        f'<text x="{x + cell / 2:.1f}" y="{y + cell * 0.68:.1f}" text-anchor="middle" font-size="9" fill="{text_color}">{int(value)}</text>'
                    )
        parts.append("</svg>")
        path.write_text("\n".join(parts) + "\n", encoding="utf-8")

    labels = list(prediction["category_labels"])
    matrix = prediction["category_confusion_matrix"]
    involved = sorted(
        {
            position
            for position in range(len(labels))
            if sum(matrix[position]) - matrix[position][position] > 0
            or sum(row[position] for row in matrix) - matrix[position][position] > 0
        }
    )
    if involved:
        subset = [[matrix[row][column] for column in involved] for row in involved]
        svg_heatmap(
            output / "category-confusion-errors.svg",
            [labels[position] for position in involved],
            subset,
            "SFT category confusion matrix (error-involved labels)",
            cell=18,
            left=230,
            top=210,
        )

    svg_heatmap(
        output / "level-confusion.svg",
        list(prediction["level_labels"]),
        prediction["level_confusion_matrix"],
        "SFT level confusion matrix",
        cell=45,
        left=70,
        top=75,
    )


def _markdown_report(report: Mapping[str, Any]) -> str:
    prediction = report["prediction_audit"]["summary"]
    overlap = report["split_overlap"]["train_vs_test"]["summary"]
    mapping = report["level_mapping"]
    slices = report["performance_slices"]
    lines = [
        "# Shougang Qwen3.5-9B LoRA SFT audit",
        "",
        "## Frozen artifacts",
        "",
        f"- Test sources: {prediction['sources']}",
        f"- Frozen artifact manifest entries: {len(report['artifact_manifest'])}",
        "- Every input is identified by file size and SHA-256 in `audit-summary.json`.",
        "",
        "## Train/test overlap",
        "",
        f"- Source-ID overlap: {overlap['source_id_overlap']}",
        f"- Exact four-field metadata overlap: {overlap['exact_metadata_overlap']}",
        f"- Normalized four-field metadata overlap: {overlap['normalized_metadata_overlap']}",
        f"- Normalized field+table overlap: {overlap['normalized_field_table_overlap']}",
        f"- Test records whose table occurs in train: {overlap['normalized_table_overlap']} / {overlap['test_sources']}",
        f"- Character-trigram near duplicates at >=0.95: {overlap['near_duplicate_at_0_95']}",
        f"- Strict-joint accuracy on seen tables: {slices['seen_table']['accuracy']:.4%} (n={slices['seen_table']['sources']})",
        f"- Strict-joint accuracy on unseen tables: {slices['unseen_table']['accuracy']:.4%} (n={slices['unseen_table']['sources']})",
        f"- Strict-joint accuracy after excluding similarity >=0.95: {slices['similarity_below_0_95']['accuracy']:.4%} (n={slices['similarity_below_0_95']['sources']})",
        "",
        "Table overlap is a group-leakage risk indicator, not proof that labels leaked.",
        "",
        "## Recomputed SFT metrics",
        "",
        f"- Stage-1 Recall@5: {prediction['stage1_recall_at_5']:.4%}",
        f"- Category accuracy / Macro-F1: {prediction['category_accuracy']:.4%} / {prediction['category_macro_f1']:.4f}",
        f"- Level accuracy / Macro-F1: {prediction['level_accuracy']:.4%} / {prediction['level_macro_f1']:.4f}",
        f"- Strict joint accuracy: {prediction['strict_joint_accuracy']:.4%}",
        f"- Stage-2 oracle accuracy when the gold label was recalled: {prediction['stage2_oracle_accuracy_when_recalled']:.4%}",
        "",
        "## Error decomposition",
        "",
        f"- Stage-1 misses: {prediction['stage1_miss']}",
        f"- Stage-2 selection errors with gold in Top-5: {prediction['stage2_selection_error']}",
        f"- Level-only errors: {prediction['level_only_error']}",
        f"- Total strict-joint errors: {prediction['sources'] - prediction['strict_joint_correct']}",
        "",
        "## Level-label diagnostic",
        "",
        f"- Train categories with more than one observed level: {mapping['ambiguous_train_categories']}",
        f"- Train-majority category-to-level rule accuracy on test: {mapping['train_majority_category_to_level_accuracy']:.4%}",
        "",
        "## Output files",
        "",
        "- `per-class-metrics.csv`: support, precision, recall, F1, and Recall@5 by category.",
        "- `category-confusion-matrix.csv`: complete category matrix.",
        "- `top-confusions.csv`: non-diagonal confusion pairs.",
        "- `error-samples.csv`: all Stage-1, Stage-2, and level errors with metadata.",
        "- `train-test-overlap.csv`: per-test overlap and nearest-train similarity.",
        "- `category-confusion-errors.svg` and `level-confusion.svg`: readable vector figures.",
        "",
    ]
    return "\n".join(lines)


def run_audit(
    *,
    train_path: Path,
    val_path: Path,
    test_path: Path,
    sft_report_path: Path,
    baseline_report_path: Path | None,
    checkpoint_metadata: Sequence[Path],
    output_dir: Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    train = load_split(train_path)
    val = load_split(val_path)
    test = load_split(test_path)
    sft_report = json.loads(sft_report_path.read_text(encoding="utf-8"))

    overlap_train_test = audit_split_overlap(train, test)
    overlap_train_val = audit_split_overlap(train, val)
    overlap_train_val_test = audit_split_overlap([*train, *val], test)
    prediction = audit_predictions(test, sft_report)
    mapping = audit_level_mapping(train, test)
    slices = performance_slices(
        overlap_train_test["per_test"], sft_report["per_source"]
    )

    artifact_paths = [
        train_path,
        val_path,
        test_path,
        sft_report_path,
        *([baseline_report_path] if baseline_report_path else []),
        *checkpoint_metadata,
    ]
    manifest = [
        {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in artifact_paths
    ]
    report: dict[str, Any] = {
        "format": "dataclassify-sft-audit-v1",
        "artifact_manifest": manifest,
        "split_sizes": {"train": len(train), "val": len(val), "test": len(test)},
        "split_overlap": {
            "train_vs_test": overlap_train_test,
            "train_vs_val": overlap_train_val,
            "train_plus_val_vs_test": overlap_train_val_test,
        },
        "prediction_audit": prediction,
        "level_mapping": mapping,
        "performance_slices": slices,
    }
    if baseline_report_path:
        baseline = json.loads(baseline_report_path.read_text(encoding="utf-8"))
        report["baseline_metrics"] = baseline.get("metrics", {})

    _write_json(output_dir / "audit-summary.json", report)
    _write_csv(output_dir / "per-class-metrics.csv", prediction["per_class"])
    _write_csv(output_dir / "per-level-metrics.csv", prediction["per_level"])
    _write_csv(output_dir / "top-confusions.csv", prediction["top_confusions"])
    _write_csv(output_dir / "error-samples.csv", prediction["errors"])
    _write_json(output_dir / "error-samples.json", prediction["errors"])
    _write_csv(
        output_dir / "train-test-overlap.csv", overlap_train_test["per_test"]
    )
    _write_matrix(
        output_dir / "category-confusion-matrix.csv",
        prediction["category_labels"],
        prediction["category_confusion_matrix"],
    )
    _write_matrix(
        output_dir / "level-confusion-matrix.csv",
        prediction["level_labels"],
        prediction["level_confusion_matrix"],
    )
    (output_dir / "audit-report.md").write_text(
        _markdown_report(report), encoding="utf-8"
    )
    _write_figures(output_dir, prediction)
    return report


def _args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--sft-report", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path)
    parser.add_argument("--checkpoint-metadata", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _args(argv)
    try:
        report = run_audit(
            train_path=args.train,
            val_path=args.val,
            test_path=args.test,
            sft_report_path=args.sft_report,
            baseline_report_path=args.baseline_report,
            checkpoint_metadata=args.checkpoint_metadata,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
        )
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"audit_sft_results: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    summary = report["prediction_audit"]["summary"]
    print(json.dumps(summary, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
