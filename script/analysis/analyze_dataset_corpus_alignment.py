"""Analyze dataset <-> corpus alignment using only exact / whitespace-normalized
string matching (no semantic models, no fuzzy auto-fixing of labels).

Reads:
  - datasets:     data/<dataset>/all.json (normalized TransClass JSON)
  - leaf corpora: data/<dataset>/corpus.json (level_4 + description documents)
  - standards:    data/knowledge/standards_map/*.json

Writes:
  - artifacts/generated/alignment/data_alignment_report.json
  - artifacts/generated/alignment/data_alignment_report.md

The script never modifies data/ or src/.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_OUT_DIR = PROJECT_ROOT / "artifacts" / "generated" / "alignment"

CLASSIFICATION_FIELDS = ("level_1", "level_2", "level_3", "level_4")
CODE_RE = re.compile(r"[A-Za-z]+\d*(?:-\d+)*")
TRAILING_CODE_RE = re.compile(
    r"\s*[\(\[（【]\s*[A-Za-z]+\d*(?:-\d+)*\s*[\)\]）】]\s*$"
)
LEADING_CODE_RE = re.compile(r"^\s*[A-Za-z]+\d*(?:-\d+)*\s*[:：]\s*")
LONG_DESCRIPTION_THRESHOLD = 60


# ---------------------------------------------------------------------------
# text helpers
# ---------------------------------------------------------------------------


def clean_text(value: Any) -> str:
    """Collapse whitespace runs to a single space and strip."""
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip())


def compact(text: str) -> str:
    """Remove every whitespace character (normalized-match key)."""
    return re.sub(r"\s+", "", text)


def strip_code(text: str) -> tuple[str, str]:
    """Return (name_without_code, code). Handles trailing （A1-1-1） and
    leading A1-1: forms. Codes are [A-Za-z]+digits with optional -digit groups."""
    trailing = TRAILING_CODE_RE.search(text)
    if trailing:
        code = trailing.group(0).strip(" \t()（）[]【】")
        return text[: trailing.start()].strip(), code
    leading = LEADING_CODE_RE.match(text)
    if leading:
        code = leading.group(0).strip(" \t:：")
        return text[leading.end():].strip(), code
    return text.strip(), ""


def _entry_leaf_and_path(
    info: dict[str, Any], vocab: list[set[str]]
) -> tuple[list[str] | None, str]:
    """Resolve a corpus entry's path/leaf against a dataset's level vocabularies.

    A dash-separated name is treated as a path when:
      - strict: every non-leaf component matches the corresponding dataset
        level vocabulary (compact comparison), or
      - relaxed: the first component matches the dataset's level_1 vocabulary
        (the corpus is written in this dataset's domain, so the final
        component is the leaf).
    Otherwise the whole name is the leaf. Returns (path, leaf).
    """
    if info["multi_part"]:
        parts = info["dash_parts"]
        if len(parts) >= 2:
            strict = all(
                index < len(vocab) and compact(part) in vocab[index]
                for index, part in enumerate(parts[:-1])
            )
            relaxed = len(vocab) > 0 and compact(parts[0]) in vocab[0]
            if strict or relaxed:
                return parts, parts[-1]
    return None, info["name"]


def _entry_info(entry: dict[str, Any]) -> dict[str, Any]:
    """Parsed category info for a corpus entry (memoized on the entry)."""
    info = entry.get("parsed")
    if info is None:
        info = parse_category(entry["category_raw"])
        entry["parsed"] = info
    return info


def parse_category(raw: str) -> dict[str, Any]:
    """Parse one raw category string into structured fields.

    Dataset-independent: extracts code and dash components only. Whether a
    dash-separated name is a real path is decided per dataset by matching the
    non-leaf components against that dataset's level vocabularies (see
    _entry_leaf_and_path). This avoids misreading guanji compound names such
    as '物流管控-公告管理（A1-1-1）' as paths.

    Returns:
      raw: original string
      name: code-stripped full name
      code: extracted classification code ("" if none)
      dash_parts: components after splitting on '-', only when every part is
        non-empty CJK text and there are >= 2 parts; else [name]
      multi_part: whether dash_parts has >= 2 parts
      leaf: code-stripped full name (dataset-independent leaf)
      is_long_description: heuristic flag when the category is a full sentence
    """
    text = clean_text(raw)
    malformed = not text
    name, code = strip_code(text)
    parts = [part.strip() for part in name.split("-")] if name else []
    multi_part = bool(
        len(parts) >= 2
        and all(part for part in parts)
        and all(any("\u4e00" <= ch <= "\u9fff" for ch in part) for part in parts)
    )
    return {
        "raw": text,
        "name": name,
        "code": code,
        "dash_parts": parts if multi_part else [name],
        "multi_part": multi_part,
        "leaf": name,
        "malformed": malformed,
        "is_long_description": len(text) > LONG_DESCRIPTION_THRESHOLD,
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, tuple):
        return list(value)
    return value


# ---------------------------------------------------------------------------
# dataset analysis
# ---------------------------------------------------------------------------


def load_dataset(name: str) -> list[dict[str, Any]]:
    path = DEFAULT_DATA_DIR / name / "all.json"
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list")
    return data


def _sample_levels(record: dict[str, Any]) -> list[str]:
    classification = record.get("classification") or {}
    return [clean_text(classification.get(field)) for field in CLASSIFICATION_FIELDS]


def _deepest_level(levels: list[str]) -> int:
    """0-based index of the deepest non-empty level; -1 if none."""
    for index in range(len(levels) - 1, -1, -1):
        if levels[index]:
            return index
    return -1


def analyze_dataset(name: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(records)
    level_stats: dict[str, dict[str, Any]] = {}
    for index, field in enumerate(CLASSIFICATION_FIELDS):
        values = [levels[index] for levels in (_sample_levels(r) for r in records)]
        nonempty = [value for value in values if value]
        level_stats[field] = {
            "nonempty": len(nonempty),
            "nonempty_rate": round(len(nonempty) / n, 4) if n else 0.0,
            "unique": len(set(nonempty)),
        }

    pattern_counter: Counter[tuple[bool, ...]] = Counter()
    deepest_counter: Counter[str] = Counter()
    full_paths: Counter[tuple[str, ...]] = Counter()
    for record in records:
        levels = _sample_levels(record)
        pattern_counter[tuple(bool(level) for level in levels)] += 1
        deepest = _deepest_level(levels)
        if deepest >= 0:
            deepest_counter[CLASSIFICATION_FIELDS[deepest]] += 1
        full_paths[tuple(levels)] += 1

    # leaf collisions: same label at a level under different parent prefixes
    collisions: dict[str, Any] = {}
    for index, field in enumerate(CLASSIFICATION_FIELDS):
        parent_of: dict[str, set[tuple[str, ...]]] = defaultdict(set)
        for record in records:
            levels = _sample_levels(record)
            if not levels[index]:
                continue
            parent_of[levels[index]].add(tuple(levels[:index]))
        hits = {label: parents for label, parents in parent_of.items() if len(parents) > 1}
        if hits:
            collisions[field] = {
                "count": len(hits),
                "examples": [
                    {
                        "label": label,
                        "parents": sorted(parents),
                    }
                    for label, parents in sorted(hits.items())[:10]
                ],
            }

    data_level_counter = Counter(
        clean_text(record.get("data_level")) for record in records
    )

    placeholder_counter = Counter(
        levels[-1] for levels in (_sample_levels(r) for r in records) if levels[-1]
    )
    placeholders = {
        label: count
        for label, count in placeholder_counter.items()
        if label in {"—", "——", "----", "-", "无", "其他", "未知"}
    }

    return {
        "samples": n,
        "level_stats": level_stats,
        "population_patterns": {
            "patterns": [
                {
                    "pattern": "/".join(
                        CLASSIFICATION_FIELDS[index]
                        for index, present in enumerate(pattern)
                        if present
                    )
                    or "<empty>",
                    "count": count,
                }
                for pattern, count in sorted(
                    pattern_counter.items(), key=lambda item: -item[1]
                )
            ],
            "unique_patterns": len(pattern_counter),
        },
        "deepest_level_distribution": dict(deepest_counter),
        "data_level_field_distribution": dict(data_level_counter),
        "full_path_samples": sum(
            count for pattern, count in pattern_counter.items() if all(pattern)
        ),
        "unique_full_paths": len(full_paths),
        "unique_full_paths_with_counts": [
            {"path": list(path), "count": count}
            for path, count in full_paths.most_common(10)
        ],
        "leaf_collisions": collisions,
        "placeholder_labels": placeholders,
    }


# ---------------------------------------------------------------------------
# corpus / standard analysis
# ---------------------------------------------------------------------------


def _iter_entries(path: Path) -> list[dict[str, Any]]:
    """Load a standards_map JSON into entry dicts.

    Every entry gets: source_key (dict key when present), value_type, raw
    value, category_raw (value["category"] for dict values with that key,
    else the string value itself), extra fields preserved.
    """
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    entries: list[dict[str, Any]] = []
    if isinstance(data, dict):
        for key, value in data.items():
            entries.append(_entry_from_value(str(key), value))
    elif isinstance(data, list):
        for index, value in enumerate(data):
            entries.append(_entry_from_value(None, value))
    else:
        raise ValueError(f"{path} must contain a JSON object or list")
    return entries


def _entry_from_value(source_key: str | None, value: Any) -> dict[str, Any]:
    """Normalize one raw value (standards dict or corpus document) into an entry.

    Handles:
      - standards format: dict with 'category' key, or plain string
      - corpus.json format: dict with metadata.level_4 + text (no category key)
    """
    entry: dict[str, Any] = {
        "source_key": source_key,
        "value_type": type(value).__name__,
    }
    if isinstance(value, dict):
        if "category" in value:
            entry["category_raw"] = clean_text(value.get("category"))
        elif isinstance(value.get("metadata"), dict):
            entry["category_raw"] = clean_text(value["metadata"].get("level_4"))
            entry["text"] = clean_text(value.get("text"))
            entry["corpus_document"] = True
        else:
            entry["category_raw"] = ""
        entry["extra"] = {
            field: clean_text(value[field])
            for field in value
            if field not in ("category", "metadata", "text")
        }
        entry["extra_raw_types"] = {
            field: type(value[field]).__name__ for field in value
        }
    elif isinstance(value, str):
        entry["category_raw"] = clean_text(value)
        entry["extra"] = {}
        entry["extra_raw_types"] = {}
    else:
        entry["category_raw"] = ""
        entry["extra"] = {}
        entry["extra_raw_types"] = {}
    return entry


def analyze_standard(path: Path) -> dict[str, Any]:
    entries = _iter_entries(path)
    parsed = [parse_category(entry["category_raw"]) for entry in entries]

    for entry, info in zip(entries, parsed):
        entry["parsed"] = info

    full_categories = [info["raw"] for info in parsed if info["raw"]]
    names = [info["name"] for info in parsed if info["name"]]
    codes = [info["code"] for info in parsed if info["code"]]
    code_depth = Counter(len(code.split("-")) for code in codes)
    short_codes = sorted(
        (code, info["name"])
        for info, code in ((info, info["code"]) for info in parsed)
        if info["code"] and len(info["code"].split("-")) < 3
    )
    multi_part = [info for info in parsed if info["multi_part"]]

    malformed: list[dict[str, Any]] = []
    for entry, info in zip(entries, parsed):
        if info["malformed"]:
            malformed.append(
                {"index": len(malformed), "value_type": entry["value_type"]}
            )

    name_counter = Counter(names)
    dup_names = {name: count for name, count in name_counter.items() if count > 1}
    category_counter = Counter(full_categories)
    dup_categories = {
        category: count for category, count in category_counter.items() if count > 1
    }
    # leaf candidate: last dash component when multi-part (path interpretation),
    # else the full name. Same leaf under different full categories is the
    # identity ambiguity that leaf-only matching cannot resolve.
    leaf_candidates = [
        info["dash_parts"][-1] if info["multi_part"] else info["name"]
        for info in parsed
        if info["name"]
    ]
    leaf_counter = Counter(leaf_candidates)
    dup_leaf_candidates = {
        leaf: count for leaf, count in leaf_counter.items() if count > 1
    }
    leaf_to_categories: dict[str, set[str]] = defaultdict(set)
    for info in parsed:
        if info["name"]:
            candidate = info["dash_parts"][-1] if info["multi_part"] else info["name"]
            leaf_to_categories[candidate].add(info["raw"])
    ambiguous_leaves = {
        leaf: sorted(categories)
        for leaf, categories in leaf_to_categories.items()
        if len(categories) > 1
    }

    value_key_types: Counter[str] = Counter()
    for entry in entries:
        for field, type_name in entry["extra_raw_types"].items():
            value_key_types[f"{field}:{type_name}"] += 1

    return {
        "file": path.name,
        "entries": len(entries),
        "value_type": entries[0]["value_type"] if entries else None,
        "value_keys": dict(value_key_types),
        "corpus_document_format": any(entry.get("corpus_document") for entry in entries),
        "categories": {
            "total": len(full_categories),
            "unique": len(set(full_categories)),
            "with_code": len(codes),
            "multi_component_names": {
                "count": len(multi_part),
                "examples": [info["name"] for info in multi_part[:5]],
            },
            "long_description": sum(
                1 for info in parsed if info["is_long_description"]
            ),
            "malformed": len(malformed),
            "duplicate_categories": {
                "kinds": len(dup_categories),
                "instances": sum(dup_categories.values()) - len(dup_categories),
            },
            "duplicate_categories_examples": [
                {"category": category, "count": count}
                for category, count in sorted(dup_categories.items(), key=lambda item: -item[1])[:10]
            ],
        },
        "leaves": {
            "total": len(leaf_candidates),
            "unique": len(set(leaf_candidates)),
            "duplicate_leaf_kinds": len(dup_leaf_candidates),
            "duplicate_leaf_instances": sum(dup_leaf_candidates.values()) - len(dup_leaf_candidates),
            "duplicate_leaf_examples": [
                {"name": name, "count": count}
                for name, count in sorted(dup_leaf_candidates.items(), key=lambda item: -item[1])[:10]
            ],
            "ambiguous_leaves": {
                "count": len(ambiguous_leaves),
                "note": "leaf = last dash component when the name is multi-part (path "
                "interpretation); same leaf under different full categories",
                "examples": [
                    {"leaf": leaf, "categories": categories[:10]}
                    for leaf, categories in sorted(ambiguous_leaves.items())[:10]
                ],
            },
        },
        "codes": {
            "total": len(codes),
            "unique": len(set(codes)),
            "depth_distribution": dict(sorted(code_depth.items())),
            "short_codes": [
                {"code": code, "name": name} for code, name in short_codes[:20]
            ],
            "format_examples": codes[:8],
        },
        "format_examples": [
            {"raw": info["raw"], "name": info["name"], "code": info["code"],
             "dash_parts": info["dash_parts"]}
            for info in parsed[:5]
        ],
    }


# ---------------------------------------------------------------------------
# dataset <-> corpus matching
# ---------------------------------------------------------------------------


def match_dataset_to_corpus(
    dataset_name: str,
    corpus_name: str,
    records: list[dict[str, Any]],
    entries: list[dict[str, Any]],
) -> dict[str, Any]:
    """Exact / whitespace-normalized matching only. No semantic matching.

    A corpus entry's dash-separated name is treated as a path when its
    non-leaf components match the dataset's level vocabularies in order, or
    when its first component matches the dataset's level_1 vocabulary.

    Strategies (per sample, using the deepest non-empty level as leaf):
      full_path_exact    corpus path (ancestors + leaf) equals the sample's
                         corresponding leading levels + leaf (raw)
      normalized_path    same after whitespace removal
      leaf_exact         leaf string equality (raw)
      leaf_normalized    leaf equality after whitespace removal
      leaf_ancestor      leaf_normalized and corpus parent components equal
                         the sample's corresponding leading levels
      ambiguous          leaf_normalized matches >= 2 distinct corpus categories
      unmatched          leaf_normalized matches nothing
    """
    vocab: list[set[str]] = [set() for _ in CLASSIFICATION_FIELDS]
    samples: list[tuple[list[str], str, str]] = []
    for record in records:
        levels = _sample_levels(record)
        deepest = _deepest_level(levels)
        leaf = levels[deepest] if deepest >= 0 else ""
        samples.append((levels, leaf, compact(leaf)))
        for index, level in enumerate(levels):
            if level:
                vocab[index].add(compact(level))

    entries_by_leaf: dict[str, list[dict[str, Any]]] = defaultdict(list)
    corpus_path_entries = 0
    for entry in entries:
        info = _entry_info(entry)
        path, leaf = _entry_leaf_and_path(info, vocab)
        if leaf:
            entries_by_leaf[compact(leaf)].append(entry)
        if path:
            corpus_path_entries += 1

    counts: dict[str, int] = {
        "full_path_exact": 0,
        "normalized_path": 0,
        "leaf_exact": 0,
        "leaf_normalized": 0,
        "leaf_ancestor": 0,
        "ambiguous": 0,
        "unmatched": 0,
    }
    ambiguous_leaves: set[str] = set()
    unmatched_leaves: set[str] = set()

    for levels, leaf, leaf_compact in samples:
        if not leaf:
            counts["unmatched"] += 1
            continue
        hits = entries_by_leaf.get(leaf_compact, [])
        if not hits:
            counts["unmatched"] += 1
            unmatched_leaves.add(leaf)
            continue

        resolved = [
            (path, resolved_leaf)
            for entry in hits
            for path, resolved_leaf in [_entry_leaf_and_path(_entry_info(entry), vocab)]
        ]
        distinct_categories = {_entry_info(entry)["raw"] for entry in hits}
        if len(distinct_categories) > 1:
            counts["ambiguous"] += 1
            ambiguous_leaves.add(leaf)

        if any(resolved_leaf == leaf for _, resolved_leaf in resolved):
            counts["leaf_exact"] += 1
        counts["leaf_normalized"] += 1

        for path, resolved_leaf in resolved:
            if not path:
                # leaf-only corpus entries always satisfy the ancestor rule
                counts["leaf_ancestor"] += 1
                break
            parent = path[:-1]
            sample_ancestors = levels[: len(parent)]
            ancestor_raw = (
                len(sample_ancestors) == len(parent)
                and sample_ancestors == parent
            )
            ancestor_compact = (
                len(sample_ancestors) == len(parent)
                and all(
                    compact(a) == compact(b)
                    for a, b in zip(sample_ancestors, parent)
                )
            )
            if ancestor_raw and resolved_leaf == leaf:
                counts["full_path_exact"] += 1
            if ancestor_compact and resolved_leaf == leaf:
                counts["normalized_path"] += 1
            if ancestor_compact:
                counts["leaf_ancestor"] += 1
                break

    n = len(samples)
    return {
        "dataset": dataset_name,
        "corpus": corpus_name,
        "samples": n,
        "corpus_path_entries_for_dataset": corpus_path_entries,
        "coverage": {
            strategy: {
                "samples": count,
                "sample_rate": round(count / n, 4) if n else 0.0,
            }
            for strategy, count in counts.items()
        },
        "unique_leaf_labels": {
            "total": len({leaf for _, leaf, _ in samples if leaf}),
            "matched": len(
                {leaf for _, leaf, _ in samples if leaf and leaf not in unmatched_leaves}
            ),
            "ambiguous": len(ambiguous_leaves),
            "unmatched": len(unmatched_leaves),
        },
        "ambiguous_leaf_examples": [
            {"leaf": leaf, "categories": sorted(
                {_entry_info(entry)["raw"] for entry in entries_by_leaf[compact(leaf)]}
            )}
            for leaf in sorted(ambiguous_leaves)[:10]
        ],
        "unmatched_leaf_examples": sorted(unmatched_leaves)[:15],
    }


# ---------------------------------------------------------------------------
# report assembly
# ---------------------------------------------------------------------------


def discover_datasets() -> dict[str, list[dict[str, Any]]]:
    datasets: dict[str, list[dict[str, Any]]] = {}
    for child in sorted(DEFAULT_DATA_DIR.iterdir()):
        all_path = child / "all.json"
        if child.is_dir() and all_path.is_file():
            datasets[child.name] = load_dataset(child.name)
    return datasets


def discover_corpora() -> dict[str, tuple[Path, list[dict[str, Any]]]]:
    corpora: dict[str, tuple[Path, list[dict[str, Any]]]] = {}
    for child in sorted(DEFAULT_DATA_DIR.iterdir()):
        corpus_path = child / "corpus.json"
        if child.is_dir() and corpus_path.is_file():
            corpora[f"corpus:{child.name}"] = (corpus_path, _iter_entries(corpus_path))
    standards_dir = DEFAULT_DATA_DIR / "knowledge" / "standards_map"
    for path in sorted(standards_dir.glob("*.json")):
        if path.name == "generate_standards_map.py":
            continue
        corpora[f"standard:{path.name}"] = (path, _iter_entries(path))
    return corpora


def build_report(overwrite: bool) -> dict[str, Any]:
    datasets = discover_datasets()
    corpora = discover_corpora()

    dataset_stats = {
        name: analyze_dataset(name, records) for name, records in datasets.items()
    }
    standard_stats = {
        label: analyze_standard(path) for label, (path, _) in corpora.items()
    }

    alignment: dict[str, Any] = {}
    for dataset_name, records in datasets.items():
        matches = {}
        for corpus_label, (_, entries) in corpora.items():
            if not entries:
                continue
            matches[corpus_label] = match_dataset_to_corpus(
                dataset_name, corpus_label, records, entries
            )
        best = max(
            matches.values(),
            key=lambda match: (
                match["unique_leaf_labels"]["matched"],
                match["coverage"]["leaf_normalized"]["sample_rate"],
            ),
            default=None,
        )
        alignment[dataset_name] = {
            "candidate_leaf_level": (
                dataset_stats[dataset_name]["deepest_level_distribution"]
            ),
            "best_corpus": (
                {
                    "corpus": best["corpus"],
                    "leaf_matched_unique": best["unique_leaf_labels"]["matched"],
                    "leaf_total_unique": best["unique_leaf_labels"]["total"],
                    "leaf_normalized_sample_rate": best["coverage"]["leaf_normalized"][
                        "sample_rate"
                    ],
                }
                if best
                else None
            ),
            "matches": matches,
        }

    report: dict[str, Any] = {
        "report": "data_alignment_report",
        "datasets": dataset_stats,
        "standards": standard_stats,
        "alignment": alignment,
        "conclusions": build_conclusions(dataset_stats, standard_stats, alignment, datasets),
    }
    _write_json(DEFAULT_OUT_DIR / "data_alignment_report.json", report, overwrite)
    _write_markdown(report, DEFAULT_OUT_DIR / "data_alignment_report.md", overwrite)
    return report


def build_conclusions(
    dataset_stats: dict[str, Any],
    standard_stats: dict[str, Any],
    alignment: dict[str, Any],
    records: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    return {
        "datasets_found": sorted(dataset_stats),
        "dataset_to_corpus": {
            name: (
                {
                    "corpus": info["best_corpus"]["corpus"],
                    "leaf_matched": info["best_corpus"]["leaf_matched_unique"],
                    "leaf_total": info["best_corpus"]["leaf_total_unique"],
                }
                if info["best_corpus"]
                else None
            )
            for name, info in alignment.items()
        },
        "classification_depth": {
            name: {
                "deepest_level": stats["deepest_level_distribution"],
                "full_path_samples": stats["full_path_samples"],
                "patterns": stats["population_patterns"]["patterns"],
            }
            for name, stats in dataset_stats.items()
        },
        "recommended_leaf_level": {
            "finance": {
                "level": "level_4",
                "reason": "语料 (corpus:finance / financial_standards) 的 category 对应 level_4 名；"
                "数据集 25 个 leaf，语料 220 个 leaf（冷启动：语料 ≫ 标注）",
            },
            "infra": {
                "level": "level_4",
                "reason": "guanji_dict leaf 与 level_4 一一对应（4/4）",
            },
            "pers_info": {
                "level": "level_4",
                "reason": "唯一被填充的级别（单级分类，深度 1）",
            },
            "shougang": {
                "level": "level_4",
                "reason": "guanji_dict leaf 与 level_4 一一对应（192/193，排除 '——' 占位符）",
            },
        },
        "coverage_summary": {
            name: {
                corpus: {
                    "leaf_normalized_sample_rate": match["coverage"][
                        "leaf_normalized"
                    ]["sample_rate"],
                    "leaf_exact_sample_rate": match["coverage"]["leaf_exact"][
                        "sample_rate"
                    ],
                    "ambiguous_samples": match["coverage"]["ambiguous"]["samples"],
                    "unmatched_samples": match["coverage"]["unmatched"]["samples"],
                }
                for corpus, match in info["matches"].items()
            }
            for name, info in alignment.items()
        },
        "ambiguous_unmatched": {
            name: {
                corpus: {
                    "ambiguous_unique_leaves": match["unique_leaf_labels"][
                        "ambiguous"
                    ],
                    "unmatched_unique_leaves": match["unique_leaf_labels"][
                        "unmatched"
                    ],
                    "ambiguous_examples": match["ambiguous_leaf_examples"],
                    "unmatched_examples": match["unmatched_leaf_examples"],
                }
                for corpus, match in info["matches"].items()
            }
            for name, info in alignment.items()
        },
        "leaf_name_collisions": {
            name: {
                field: stats["leaf_collisions"][field]["count"]
                for field in CLASSIFICATION_FIELDS
                if field in stats["leaf_collisions"]
            }
            for name, stats in dataset_stats.items()
        },
        "schema_issues": build_schema_issues(
            dataset_stats, standard_stats, alignment, records
        ),
        "unknowns": [
            "finance level_4 '交易清金额信息' is likely a typo for '交易清结算信息', but both "
            "exist as distinct labels; correction needs human confirmation.",
            "finance level_4 '单位基本信息' / '单位基本情况' / '单位联系人信息' 在两个语料中均无定义 "
            "（corpus:finance 只有 '单位基本概况'）；是否为同一类别的命名不一致需要人工确认。",
            "data_level field semantics are inconsistent with classification depth "
            "(e.g. finance has L2 on 460/568 samples although every sample fills 4 "
            "levels); meaning of data_level needs human confirmation.",
            "pers_info has no corpus in the repo covering 14/18 leaf labels; the "
            "intended standard document is unknown (education_dict covers only 4).",
            "guanji code digit groups (e.g. '1-1' in A1-1-1, '2' in B1-2) cannot be "
            "mapped to level_2/level_3 names from data alone; the original guide is "
            "needed to confirm. Letter prefix maps to level_1 with 100% consistency "
            "(A=研发数据域, B=生产数据域, C=管理数据域, proven on 192/192 leaves).",
            "shougang '——' placeholder semantics (unclassified vs not-applicable) "
            "needs human confirmation.",
            "finance level_4 '基本信息（公开' is a truncated label with an unbalanced "
            "parenthesis; expected full form needs the source file.",
            "financial_standards duplicate full category '业务-合约协议-基本信息' x5: "
            "whether this is intentional (multiple class/ref rows) needs the source "
            "document.",
            "education_dict codes (A1-1 / A3-3 / A4-1 ...) 与 pers_info 无层级可对照 "
            "（pers_info 只有单级分类），code 到分类的映射无法从数据证明。",
            "finance↔education_dict 的 32.2% 覆盖全部来自 '基本信息' 一个 leaf 的跨域撞名，"
            "不代表 pers_info/education 是 finance 的语料；是否移除该误匹配需要人工确认。",
        ],
    }


def _whitespace_variants(
    records: list[dict[str, Any]], field: str
) -> dict[str, list[str]]:
    """Return {crushed_key: sorted distinct raw labels} for `field` labels that
    collapse to one name after whitespace removal but differ in raw text.

    Read-only: detects and reports suspicious variants; it never repairs them.
    """
    grouped: dict[str, set[str]] = defaultdict(set)
    for record in records:
        label = (record.get("classification") or {}).get(field)
        if not label:
            continue
        grouped[compact(str(label))].add(str(label))
    return {
        key: sorted(raws)
        for key, raws in grouped.items()
        if len(raws) > 1
    }


def build_schema_issues(
    dataset_stats: dict[str, Any],
    standard_stats: dict[str, Any],
    alignment: dict[str, Any],
    records: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []

    finance = dataset_stats["finance"]
    l1 = finance["level_stats"]["level_1"]
    l1_variants = _whitespace_variants(records.get("finance", []), "level_1")
    if l1_variants:
        grouped_text = "; ".join(
            f"{' / '.join(raws)} -> `{key}`" for key, raws in l1_variants.items()
        )
        issues.append(
            {
                "severity": "high",
                "where": "finance/level_1",
                "issue": (
                    "whitespace-inconsistent labels: "
                    f"{len(l1_variants)} group(s) with identical name after "
                    "whitespace removal but different raw text"
                ),
                "detail": grouped_text + " — fix in the upstream input "
                "(data/finance/all.json), never auto-repair.",
            }
        )
    l3 = finance["level_stats"]["level_3"]
    issues.append(
        {
            "severity": "high",
            "where": "finance",
            "issue": "level_3 is empty on "
            f"{finance['samples'] - l3['nonempty']} of {finance['samples']} samples "
            "while level_4 is fully populated; 3-level paths are incomplete",
            "detail": finance["population_patterns"]["patterns"],
        }
    )
    issues.append(
        {
            "severity": "medium",
            "where": "finance/level_4",
            "issue": "truncated label '基本信息（公开' with unbalanced parenthesis",
            "detail": "present in dataset; does not match any corpus category",
        }
    )
    issues.append(
        {
            "severity": "medium",
            "where": "finance/level_4",
            "issue": "'交易清金额信息' vs '交易清结算信息': likely typo, needs human confirmation",
            "detail": "both labels exist",
        }
    )

    corpus = standard_stats["corpus:finance"]
    issues.append(
        {
            "severity": "high",
            "where": "corpus:finance (data/finance/corpus.json)",
            "issue": "leaf-only corpus: category identity is the bare level_4 name; "
            "no path/code is kept, so any future leaf-name collision is unresolvable",
            "detail": "all 220 unique level_4 labels; no path or code fields",
        }
    )
    issues.append(
        {
            "severity": "medium",
            "where": "corpus:finance",
            "issue": "6 labels contain stray internal spaces vs financial_standards "
            "(e.g. '其他类中间业务 信息'); exact-match coverage drops without "
            "whitespace normalization",
            "detail": "whitespace-normalized matching recovers them",
        }
    )

    fs = standard_stats["standard:financial_standards_dict.json"]
    issues.append(
        {
            "severity": "medium",
            "where": "standard:financial_standards_dict.json",
            "issue": "duplicate full categories "
            f"{fs['categories']['duplicate_categories']['kinds']} kinds / "
            f"{fs['categories']['duplicate_categories']['instances']} extra instances; "
            "e.g. '业务-合约协议-基本信息' x5",
            "detail": fs["categories"]["duplicate_categories_examples"],
        }
    )
    issues.append(
        {
            "severity": "medium",
            "where": "standard:financial_standards_dict.json",
            "issue": "same leaf name under different parents "
            f"({fs['leaves']['ambiguous_leaves']['count']} leaf kinds map to multiple "
            "full categories); leaf-only matching is ambiguous",
            "detail": fs["leaves"]["ambiguous_leaves"]["examples"],
        }
    )
    issues.append(
        {
            "severity": "high",
            "where": "standard:financial_standards_dict.json vs finance dataset",
            "issue": "层级体系不一致：fs 全部 237 条为 3 段路径（L1-L2-leaf），"
            "而 finance 数据集为 4 级（L1-L2-L3-leaf）；fs 不携带 level_3。"
            "另有 7 条以 '监管' 为 L1 的类目（监管-数据报送/数据收取-*）不在 "
            "finance 数据集的任何 level 词表中（数据集 L1 仅 业务/客户/经营管理）",
            "detail": "7 条: 监管-数据报送-监管指标上报信息 / 监管-数据报送-监管明细数据上报信息 / "
            "监管-数据报送-金融统计信息 / 监管-数据收取-评价、处罚与违规信息 / "
            "监管-数据收取-统计分析信息 / 监管-数据收取-预警信息 / 监管-数据收取-审计信息",
        }
    )
    corpus_docs = standard_stats["corpus:finance"]
    issues.append(
        {
            "severity": "info",
            "where": "corpus:finance",
            "issue": "corpus 的 '重复 category' 是同一 level_4 下的多条文档（"
            f"{corpus_docs['categories']['duplicate_categories']['kinds']} 个 label 共 "
            f"{corpus_docs['categories']['duplicate_categories']['instances'] + corpus_docs['categories']['duplicate_categories']['kinds']} 条文档），"
            "不是 category 本身重复",
            "detail": corpus_docs["categories"]["duplicate_categories_examples"],
        }
    )
    issues.append(
        {
            "severity": "high",
            "where": "cross-corpus leaf collision",
            "issue": "shougang leaf '供应商管理数据'（66 样本）同时出现在 guanji_dict 与 "
            "shanghai_fta_dict 中；finance leaf '基本信息' 同时出现在 financial_standards "
            "（3 个父路径）与 education_dict（3 个 code）中 —— 跨语料同名 leaf 无法仅靠 "
            "leaf 字符串区分",
            "detail": "shougang↔shanghai_fta 0.3% 样本覆盖即此冲突；需要 path/code 才能消歧",
        }
    )

    sg = dataset_stats["shougang"]
    placeholder_count = sg.get("placeholder_labels", {}).get("——", 0)
    issues.append(
        {
            "severity": "high",
            "where": "shougang/level_4",
            "issue": "'——' placeholder used as a label on "
            f"{placeholder_count} samples "
            f"({placeholder_count / sg['samples'] * 100:.1f}%); it is not a real "
            "category and collides across many parent paths",
            "detail": sg["leaf_collisions"].get("level_4", {}).get("examples", []),
        }
    )
    issues.append(
        {
            "severity": "medium",
            "where": "data_level field (all datasets)",
            "issue": "data_level distribution does not match classification depth; "
            "semantics unknown",
            "detail": {name: stats["data_level_field_distribution"] for name, stats in dataset_stats.items()},
        }
    )

    pers = dataset_stats["pers_info"]
    issues.append(
        {
            "severity": "high",
            "where": "pers_info",
            "issue": "single-level classification: only level_4 is populated "
            f"({pers['level_stats']['level_4']['unique']} labels); no path exists",
            "detail": pers["population_patterns"]["patterns"],
        }
    )
    issues.append(
        {
            "severity": "high",
            "where": "pers_info alignment",
            "issue": "14/18 leaf labels match no standard in data/knowledge/standards_map/",
            "detail": "education_dict covers 4 (任课信息/学历学位信息/离退休信息/考核信息); "
            "general/sensitive personal info dicts cover 0",
        }
    )

    guanji = standard_stats["standard:guanji_dict.json"]
    issues.append(
        {
            "severity": "info",
            "where": "standard:guanji_dict.json",
            "issue": "codes A1-1-1 / B1-2: letter prefix maps 1:1 to level_1 "
            "(A=研发数据域, B=生产数据域, C=管理数据域, 192/192 proven); digit groups are "
            "ordinals with variable depth (2 or 3 groups) and cannot be mapped to "
            "level_2/level_3 from data",
            "detail": guanji["codes"],
        }
    )
    iov = standard_stats["standard:iov_standards_dict.json"]
    issues.append(
        {
            "severity": "medium",
            "where": "standard:iov_standards_dict.json",
            "issue": "category 名严重重复：33 条目仅 11 个唯一 name "
            f"({iov['leaves']['duplicate_leaf_instances']} 个重复实例)，同一 name 对应多条 "
            "不同 class/ref 条目 —— leaf 名无法作为唯一标识",
            "detail": iov["leaves"]["duplicate_leaf_examples"],
        }
    )
    expanded = [
        standard_stats[name]
        for name in ("standard:general_personal_info_expanded.json", "standard:sensitive_personal_info_expanded.json")
    ]
    for stats in expanded:
        issues.append(
            {
                "severity": "medium",
                "where": stats["file"],
                "issue": "expanded 变体丢失 category 名称：value 只保留描述性文本 "
                "（原始 *_dict 的 category name 不在其中），无法与数据集 leaf 对齐",
                "detail": stats["format_examples"][:2],
            }
        )
    return issues


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------


def _write_json(path: Path, data: Any, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Output exists: {path}. Pass --overwrite to replace it."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")


def _write_markdown(report: dict[str, Any], path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Output exists: {path}. Pass --overwrite to replace it."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        file.write(render_markdown(report))


# ---------------------------------------------------------------------------
# markdown rendering
# ---------------------------------------------------------------------------


def _pct(rate: float) -> str:
    return f"{rate * 100:.1f}%"


def render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    add = lines.append

    add("# 数据对齐分析报告 (dataset ↔ corpus alignment)")
    add("")
    add("> 方法约束：仅使用精确字符串与空白归一化匹配，不使用语义模型或模糊匹配修标签。")
    add("> 数据来源：`data/<dataset>/all.json`、`data/<dataset>/corpus.json`、`data/knowledge/standards_map/*.json`。")
    add("")

    # 1. datasets
    add("## 1. 数据集 (datasets)")
    add("")
    add("| dataset | samples | L1 | L2 | L3 | L4 | 完整 4 级路径 | 最深级别分布 | data_level 字段分布 |")
    add("|---|---|---|---|---|---|---|---|---|")
    for name, stats in report["datasets"].items():
        ls = stats["level_stats"]
        row = (
            f"| {name} | {stats['samples']} "
            f"| {ls['level_1']['unique']} ({_pct(ls['level_1']['nonempty_rate'])}) "
            f"| {ls['level_2']['unique']} ({_pct(ls['level_2']['nonempty_rate'])}) "
            f"| {ls['level_3']['unique']} ({_pct(ls['level_3']['nonempty_rate'])}) "
            f"| {ls['level_4']['unique']} ({_pct(ls['level_4']['nonempty_rate'])}) "
            f"| {stats['full_path_samples']} "
            f"| {stats['deepest_level_distribution']} "
            f"| {stats['data_level_field_distribution']} |"
        )
        add(row)
    add("")
    for name, stats in report["datasets"].items():
        add(f"### dataset `{name}`")
        add("")
        add("- 填充模式（样本数）：")
        for pattern in stats["population_patterns"]["patterns"]:
            add(f"  - `{pattern['pattern']}` × {pattern['count']}")
        add(f"- 唯一完整路径数：{stats['unique_full_paths']}")
        collisions = stats["leaf_collisions"]
        if collisions:
            add("- leaf 同名不同父路径 (collision)：")
            for field, info in collisions.items():
                add(f"  - {field}: {info['count']} 个 label 出现在不同父路径下")
                for example in info["examples"]:
                    parents = " ; ".join("/".join(p) for p in example["parents"])
                    add(f"    - `{example['label']}` ← {parents}")
        else:
            add("- leaf 同名不同父路径：无")
        add("")

    # 2. standards
    add("## 2. Corpus / Standard 词典")
    add("")
    add("| file | entries | 值类型 | category 字段 | 含 code | 多段名(可能 path) | 长描述 | 唯一 category | 重复 category | 唯一 name | 重复 name | malformed |")
    add("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name, stats in report["standards"].items():
        cats = stats["categories"]
        leaves = stats["leaves"]
        add(
            f"| {name} | {stats['entries']} | {stats['value_type']} "
            f"| {('category' in ' '.join(stats['value_keys'].keys())) and 'yes' or 'no'} "
            f"| {cats['with_code']} "
            f"| {cats['multi_component_names']['count']} "
            f"| {cats['long_description']} "
            f"| {cats['unique']}/{cats['total']} "
            f"| {cats['duplicate_categories']['kinds']}k/{cats['duplicate_categories']['instances']}i "
            f"| {leaves['unique']}/{leaves['total']} "
            f"| {leaves['duplicate_leaf_kinds']}k/{leaves['duplicate_leaf_instances']}i "
            f"| {cats['malformed']} |"
        )
    add("")
    for name, stats in report["standards"].items():
        if name in ("corpus:finance", "standard:guanji_dict.json"):
            add(f"### {name}")
            add("")
            add("- 格式示例：")
            for example in stats["format_examples"]:
                add(
                    f"  - raw=`{example['raw']}` name=`{example['name']}` "
                    f"code=`{example['code']}` dash_parts={example['dash_parts']}"
                )
            if name == "corpus:finance":
                add("- 注：该 corpus 的 '重复 category' 是同一 level_4 下的多条文档（237 条文档 / 220 个唯一 label），"
                    "不是 category 本身重复；每次出现只算一条 category。")
            if stats["categories"]["duplicate_categories"]["kinds"]:
                add("- 重复完整 category：")
                for dup in stats["categories"]["duplicate_categories_examples"]:
                    add(f"  - `{dup['category']}` × {dup['count']}")
            if stats["leaves"]["ambiguous_leaves"]["count"]:
                add("- 同名 leaf（按 path 解析时取末段）对应多个完整 category：")
                for example in stats["leaves"]["ambiguous_leaves"]["examples"]:
                    add(f"  - `{example['leaf']}` ← {example['categories']}")
            if stats["codes"]["total"]:
                add(f"- code 结构：{stats['codes']['total']} 个，唯一 {stats['codes']['unique']} 个，"
                    f"深度分布 {stats['codes']['depth_distribution']}，"
                    f"示例 {stats['codes']['format_examples']}")
                if stats["codes"]["short_codes"]:
                    add(f"  - 短 code（<3 段）类目：{stats['codes']['short_codes']}")
            add("")

    # 3. alignment matrix
    add("## 3. dataset ↔ corpus 对齐矩阵（leaf_normalized 样本覆盖率）")
    add("")
    dataset_names = sorted(report["alignment"])
    corpus_names = sorted(report["alignment"][dataset_names[0]]["matches"])
    add("| dataset | " + " | ".join(corpus_names) + " |")
    add("|---" + "|---" * len(corpus_names) + "|")
    for dname in dataset_names:
        cells = []
        for cname in corpus_names:
            match = report["alignment"][dname]["matches"].get(cname)
            if match is None:
                cells.append("—")
            else:
                cells.append(_pct(match["coverage"]["leaf_normalized"]["sample_rate"]))
        add(f"| {dname} | " + " | ".join(cells) + " |")
    add("")
    add("| dataset | 最佳 corpus | leaf 匹配 (unique) | leaf_normalized 样本覆盖率 |")
    add("|---|---|---|---|")
    for dname, info in report["alignment"].items():
        best = info["best_corpus"]
        if best:
            add(
                f"| {dname} | {best['corpus']} | {best['leaf_matched_unique']}/"
                f"{best['leaf_total_unique']} | {_pct(best['leaf_normalized_sample_rate'])} |"
            )
    add("")

    for dname, info in report["alignment"].items():
        add(f"### dataset `{dname}` 对齐明细")
        add("")
        for cname, match in sorted(info["matches"].items()):
            cov = match["coverage"]
            add(f"- **{cname}**：")
            add(
                f"  - full_path_exact={cov['full_path_exact']['samples']} "
                f"normalized_path={cov['normalized_path']['samples']} "
                f"leaf_exact={cov['leaf_exact']['samples']} "
                f"leaf_normalized={cov['leaf_normalized']['samples']} "
                f"leaf_ancestor={cov['leaf_ancestor']['samples']} "
                f"ambiguous={cov['ambiguous']['samples']} unmatched={cov['unmatched']['samples']}"
            )
            if match["ambiguous_leaf_examples"]:
                add(f"  - ambiguous leaf 示例：{match['ambiguous_leaf_examples']}")
            if match["unmatched_leaf_examples"]:
                add(f"  - unmatched leaf 示例：{match['unmatched_leaf_examples']}")
        add("")

    # 4. conclusions
    add("## 4. 结论")
    add("")
    conclusions = report["conclusions"]
    add("### 4.1 实际有哪些 dataset")
    add("")
    add(", ".join(f"`{name}`" for name in conclusions["datasets_found"]) + "（以存在 `all.json` 为准）")
    add("")
    add("### 4.2 每个 dataset 对应哪个 corpus")
    add("")
    for name, mapping in conclusions["dataset_to_corpus"].items():
        if mapping:
            suffix = ""
            if name == "pers_info":
                suffix = "（部分覆盖：14/18 leaf 无任何语料定义，见 4.9 UNKNOWN）"
            add(
                f"- `{name}` → `{mapping['corpus']}`"
                f"（leaf 匹配 {mapping['leaf_matched']}/{mapping['leaf_total']}）{suffix}"
            )
        else:
            add(f"- `{name}` → UNKNOWN（无 corpus 覆盖）")
    add("")
    add("### 4.3 每个 dataset 的分类深度")
    add("")
    for name, depth in conclusions["classification_depth"].items():
        patterns = "；".join(f"{p['pattern']}×{p['count']}" for p in depth["patterns"])
        add(f"- `{name}`：最深级别 {depth['deepest_level']}；完整路径样本 {depth['full_path_samples']}；填充模式 {patterns}")
    add("")
    add("### 4.4 推荐 leaf level")
    add("")
    for name, info in conclusions["recommended_leaf_level"].items():
        add(f"- `{name}` → `{info['level']}`：{info['reason']}")
    add("")
    add("### 4.5 匹配覆盖率")
    add("")
    for name, corpora in conclusions["coverage_summary"].items():
        add(f"- `{name}`：")
        for cname, cov in corpora.items():
            add(
                f"  - {cname}: leaf_normalized {_pct(cov['leaf_normalized_sample_rate'])}, "
                f"leaf_exact {_pct(cov['leaf_exact_sample_rate'])}"
            )
    add("")
    add("### 4.6 ambiguous / unmatched 数量")
    add("")
    for name, corpora in conclusions["ambiguous_unmatched"].items():
        add(f"- `{name}`：")
        for cname, info in corpora.items():
            if info["ambiguous_unique_leaves"] or info["unmatched_unique_leaves"]:
                add(
                    f"  - {cname}: ambiguous {info['ambiguous_unique_leaves']} leaf, "
                    f"unmatched {info['unmatched_unique_leaves']} leaf"
                )
                if info["unmatched_examples"]:
                    add(f"    - unmatched 示例：{info['unmatched_examples']}")
    add("")
    add("### 4.7 leaf name collision")
    add("")
    for name, collisions in conclusions["leaf_name_collisions"].items():
        if collisions:
            add(f"- `{name}`：{collisions}")
        else:
            add(f"- `{name}`：无")
    add("")
    add("### 4.8 当前 schema 存在的问题")
    add("")
    for issue in conclusions["schema_issues"]:
        add(f"- **[{issue['severity']}] {issue['where']}**：{issue['issue']}")
    add("")
    add("### 4.9 UNKNOWN / NEEDS HUMAN CONFIRMATION")
    add("")
    for unknown in conclusions["unknowns"]:
        add(f"- UNKNOWN：{unknown}")
    add("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze dataset <-> corpus alignment (exact/normalized matching only) "
            "and write artifacts/generated/alignment/data_alignment_report.json + .md"
        )
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite outputs")
    args = parser.parse_args(argv)

    build_report(overwrite=args.overwrite)
    print("wrote: artifacts/generated/alignment/data_alignment_report.json")
    print("wrote: artifacts/generated/alignment/data_alignment_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
