"""Convert tabular metadata into the normalized classification JSON format."""

from __future__ import annotations

import json
import os
import re
import tempfile
import uuid
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence


STANDARD_FIELDS = (
    "database_name",
    "database_description",
    "table_name",
    "table_description",
    "field_name",
    "field_description",
    "field_type",
    "level_1",
    "level_2",
    "level_3",
    "level_4",
    "data_level",
)
CLASSIFICATION_FIELDS = ("level_1", "level_2", "level_3", "level_4")
IDENTITY_FIELDS = ("database_name", "table_name", "field_name")

# Matches a trailing classification code such as （A3）, (A01), （A1-1-3）,
# or 【A】 while preserving semantic parentheses.
_TRAILING_CODE_RE = re.compile(
    r"[\(\[（【]\s*([A-Za-z]+\d*(?:-\d+)*)\s*[\)\]）】]\s*$"
)


@lru_cache(maxsize=1)
def _load_pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(
            "Tabular preprocessing requires pandas. For .xlsx files, "
            "openpyxl is also required."
        ) from exc
    return pd


def _validated_file(path: str | Path, description: str) -> Path:
    result = Path(path).expanduser().resolve()
    if not result.is_file():
        raise FileNotFoundError(f"{description} not found: {result}")
    return result


def read_data(path: str | Path):
    pd = _load_pandas()
    source = _validated_file(path, "Input file")
    suffix = source.suffix.lower()

    if suffix == ".xlsx":
        frame = pd.read_excel(source)
    elif suffix == ".csv":
        frame = pd.read_csv(source)
    else:
        raise ValueError(
            f"Unsupported input type '{suffix}'. Expected .csv or .xlsx."
        )

    frame.columns = [str(column).strip() for column in frame.columns]
    if frame.columns.duplicated().any():
        duplicates = sorted(set(frame.columns[frame.columns.duplicated()]))
        raise ValueError("Duplicate source columns: " + ", ".join(duplicates))
    return frame


def load_mapping(path: str | Path) -> dict[str, str]:
    mapping_path = _validated_file(path, "Mapping file")
    try:
        with mapping_path.open("r", encoding="utf-8-sig") as file:
            mapping = json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid mapping JSON: {mapping_path}: {exc}") from exc

    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("Mapping JSON must be a non-empty object")
    if not all(isinstance(key, str) and isinstance(value, str) for key, value in mapping.items()):
        raise ValueError("Every mapping key and value must be a string")

    normalized: dict[str, str] = {}
    for raw_key, raw_value in mapping.items():
        key = raw_key.strip()
        value = raw_value.strip()
        if not key or not value:
            raise ValueError("Mapping keys and values cannot be empty")
        if key in normalized:
            raise ValueError(f"Duplicate mapping source after trimming: {key!r}")
        normalized[key] = value
    return normalized


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        missing = _load_pandas().isna(value)
        if bool(missing):
            return ""
    except (TypeError, ValueError):
        # Non-scalar values are not expected here; stringify them below so
        # validation can still report the relevant source field.
        pass
    return re.sub(r"\s+", " ", str(value).strip())


def detect_trailing_code(text: str) -> str:
    """Return the trailing classification code of *text* ("" when absent).

    Detection never mutates the value; callers decide via
    ``strip_trailing_codes`` whether stripping is allowed.
    """
    match = _TRAILING_CODE_RE.search(text)
    return match.group(1) if match else ""


def normalize_label(value: Any) -> str:
    """Whitespace-normalize a label and strip a trailing classification code.

    Kept for callers that explicitly opt into stripping (legacy behavior);
    the canonical preprocessing path uses :func:`clean_label` which records
    detected codes instead of silently removing them.
    """
    text = clean_text(value)
    return _TRAILING_CODE_RE.sub("", text).strip()


def clean_label(value: Any) -> tuple[str, str]:
    """Whitespace-normalize one label without altering its semantics.

    Returns ``(label, trailing_code)`` where ``trailing_code`` is "" when the
    label carries no bracket-style suffix. The label text itself is never
    truncated here.
    """
    text = clean_text(value)
    return text, detect_trailing_code(text)


def apply_rewrite_rules(
    value: str,
    rules: Sequence[Mapping[str, str]] | None,
) -> tuple[str, str | None]:
    """Apply explicit label-rewrite rules (exact match on the whole value).

    Returns ``(value, rule_description)``; ``rule_description`` is None when
    no rule fired. Rules make historical hand-edits (e.g. whitespace fixes
    such as '经营 管理' -> '经营管理') reproducible and auditable instead of
    editing data files in place.
    """
    if not rules:
        return value, None
    for rule in rules:
        match_value = str(rule.get("match", ""))
        replacement = rule.get("replace")
        field = str(rule.get("field", ""))
        if value == match_value and replacement is not None:
            description = ": ".join(part for part in (field, match_value) if part)
            return str(replacement), f"{description} -> {replacement}"
    return value, None


def normalize_name(value: Any) -> str:
    text = clean_text(value).lower()
    text = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def normalize_type(value: Any) -> str:
    text = clean_text(value).lower()
    if not text:
        return ""

    base_type = re.split(r"[\s(]", text, maxsplit=1)[0]
    mapping = {
        "varchar": "STRING", "varchar2": "STRING", "nvarchar": "STRING",
        "nvarchar2": "STRING", "string": "STRING", "char": "STRING",
        "nchar": "STRING", "text": "STRING", "clob": "STRING",
        "tinyint": "INTEGER", "smallint": "INTEGER", "mediumint": "INTEGER",
        "int": "INTEGER", "integer": "INTEGER", "bigint": "INTEGER",
        "decimal": "DECIMAL", "numeric": "DECIMAL", "number": "DECIMAL",
        "float": "FLOAT", "real": "FLOAT", "double": "FLOAT",
        "bool": "BOOLEAN", "boolean": "BOOLEAN", "datetime": "DATETIME",
        "timestamp": "DATETIME", "date": "DATE", "time": "TIME",
        "binary": "BINARY", "varbinary": "BINARY", "blob": "BINARY",
        "json": "JSON",
    }
    return mapping.get(base_type, text.upper())


def normalize_level(value: Any) -> str:
    text = clean_text(value).upper()
    if not text:
        return ""
    aliases = {
        "1": "L1", "2": "L2", "3": "L3", "4": "L4",
        "LEVEL1": "L1", "LEVEL2": "L2", "LEVEL3": "L3", "LEVEL4": "L4",
    }
    normalized = aliases.get(text, text)
    if normalized not in {"L1", "L2", "L3", "L4"}:
        raise ValueError(f"Unsupported data level: {value!r}")
    return normalized


def convert_schema(frame, mapping: dict[str, str]):
    invalid_targets = sorted(set(mapping.values()) - set(STANDARD_FIELDS))
    if invalid_targets:
        raise ValueError("Unsupported mapping targets: " + ", ".join(invalid_targets))

    duplicate_targets = sorted(
        target for target in set(mapping.values())
        if list(mapping.values()).count(target) > 1
    )
    if duplicate_targets:
        raise ValueError("Multiple source columns map to: " + ", ".join(duplicate_targets))

    missing_sources = sorted(source for source in mapping if source not in frame.columns)
    if missing_sources:
        raise ValueError("Mapped source columns not found: " + ", ".join(missing_sources))

    result = frame.rename(columns=mapping)
    for field in STANDARD_FIELDS:
        if field not in result.columns:
            result[field] = ""
    return result[list(STANDARD_FIELDS)].copy()


def _atomic_write_json(data: Any, path: Path, overwrite: bool) -> None:
    path = path.expanduser().resolve()
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {path}. Pass --overwrite to replace it."
        )
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


def preprocess(
    input_file: str | Path,
    mapping_file: str | Path,
    output_file: str | Path,
    *,
    dataset: str,
    overwrite: bool = False,
    missing_field_policy: str = "error",
    strip_trailing_codes: bool = False,
    rewrite_rules: Sequence[Mapping[str, str]] | None = None,
) -> list[dict[str, Any]]:
    """Preprocess one CSV/XLSX file and write normalized JSON.

    ``dataset`` participates in the stable sample id (uuid5 over
    dataset + database/table/field), so renaming the input file never
    changes identities. Label semantics are preserved by default:
    trailing bracket-style codes are detected and reported per record
    (``label_notes``) but only stripped when ``strip_trailing_codes`` is
    set. ``rewrite_rules`` applies explicit, auditable label fixes and
    records them under ``rewritten_from``.
    """
    if not str(dataset).strip():
        raise ValueError("dataset must be a non-empty name")
    if missing_field_policy not in {"error", "skip"}:
        raise ValueError("missing_field_policy must be 'error' or 'skip'")

    source = _validated_file(input_file, "Input file")
    frame = convert_schema(read_data(source), load_mapping(mapping_file))

    if frame.empty:
        raise ValueError("Input dataset is empty")
    for column in frame.columns:
        frame[column] = frame[column].map(clean_text)

    empty_rows = frame.index[frame["field_name"] == ""].tolist()
    if empty_rows:
        examples = ", ".join(str(index + 2) for index in empty_rows[:10])
        suffix = "..." if len(empty_rows) > 10 else ""
        message = f"field_name is empty at source rows: {examples}{suffix}"
        if missing_field_policy == "error":
            raise ValueError(message)
        warnings.warn(
            f"Skipping {len(empty_rows)} row(s): {message}",
            UserWarning,
            stacklevel=2,
        )
        frame = frame.drop(index=empty_rows).copy()
        if frame.empty:
            raise ValueError("No valid rows remain after skipping empty field_name values")

    # Label pipeline per classification column. Helper columns (``__code__`` /
    # ``__rule__`` / ``__orig__``) keep per-row audit data aligned through the
    # later drop_duplicates and are removed before record construction.
    for column in CLASSIFICATION_FIELDS:
        new_labels: list[str] = []
        codes: list[str] = []
        rule_notes: list[str] = []
        originals: list[str] = []
        for raw in frame[column]:
            original, _ = clean_label(raw)
            value, note = apply_rewrite_rules(original, rewrite_rules)
            code = detect_trailing_code(value)
            if strip_trailing_codes:
                # legacy behavior: remove the suffix from the stored label
                value = _TRAILING_CODE_RE.sub("", value).strip()
                code = ""
            new_labels.append(value)
            codes.append(code)
            rule_notes.append(note or "")
            originals.append(original)
        frame[column] = new_labels
        frame[f"__code__{column}"] = codes
        frame[f"__rule__{column}"] = rule_notes
        frame[f"__orig__{column}"] = originals
    frame["field_type"] = frame["field_type"].map(normalize_type)
    frame["data_level"] = frame["data_level"].map(normalize_level)

    conflicts: list[tuple[Any, ...]] = []
    for identity, group in frame.groupby(list(IDENTITY_FIELDS), dropna=False, sort=False):
        labels = group[list(CLASSIFICATION_FIELDS) + ["data_level"]].drop_duplicates()
        if len(labels) > 1:
            conflicts.append(identity)
    if conflicts:
        examples = ", ".join(
            "/".join(value or "<empty>" for value in identity)
            for identity in conflicts[:10]
        )
        raise ValueError(f"Conflicting labels found for the same field: {examples}")

    frame = frame.drop_duplicates(subset=list(IDENTITY_FIELDS), keep="first")
    result: list[dict[str, Any]] = []
    records_out = frame.to_dict(orient="records")
    for row in records_out:
        identity = "\x1f".join(
            [str(dataset).strip(), *(row[field] for field in IDENTITY_FIELDS)]
        )
        record_id = str(uuid.uuid5(uuid.NAMESPACE_URL, identity))
        notes = [
            {"field": column, "kept_code": row[f"__code__{column}"]}
            for column in CLASSIFICATION_FIELDS
            if row[f"__code__{column}"]
        ]
        rewritten_from = {
            column: row[f"__orig__{column}"]
            for column in CLASSIFICATION_FIELDS
            if row[f"__rule__{column}"]
        }
        entry: dict[str, Any] = {
            "id": record_id,
            "key": normalize_name(row["field_name"]),
            "label_status": (
                "labeled"
                if any(row[field] for field in CLASSIFICATION_FIELDS)
                else "unlabeled"
            ),
            "metadata": {
                "database_name": row["database_name"],
                "database_description": row["database_description"],
                "table_name": row["table_name"],
                "table_description": row["table_description"],
                "field_name": row["field_name"],
                "field_description": row["field_description"],
                "field_type": row["field_type"],
                "value": "",
            },
            "classification": {field: row[field] for field in CLASSIFICATION_FIELDS},
            "data_level": row["data_level"],
        }
        if notes:
            entry["label_notes"] = notes
        if rewritten_from:
            entry["rewritten_from"] = rewritten_from
        result.append(entry)

    _atomic_write_json(result, Path(output_file), overwrite)
    return result
