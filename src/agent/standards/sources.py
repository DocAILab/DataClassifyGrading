"""Raw standard readers (Phase 1): read the ORIGINAL standard workbooks into
plain raw-entry dicts.

The canonical standard must be built from the original source workbooks
(data/raw), not from the already-compressed standards_map JSON digests
(financial_standards_dict.json / guanji_dict.json dropped real path depth and
grading columns). These readers only extract, never normalize semantics:
grading columns are kept as raw strings.

Excel layout notes (verified against data/raw at 2026-08-20):
- finance 金融行业数据安全分类分级标准指南.xlsx / sheet "Table 1":
  columns 一级子类|二级子类|二级定义|三级子类|三级定义|四级子类|内容|安全级别|备注|部门意见;
  merged 一级/二级/三级 subclasses carry forward; "安全级别" header row is
  "最低安全级别参考".
- shougang 关基-数据分类分级目录.xlsx / sheet "数据分类分级":
  columns 一级分类|一级定义|二级分类|二级定义|三级分类|三级定义|四级分类|四级定义|
  数据资源说明（内容）|分级|数据资源; merged 一级/二级/三级 carry forward.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from agent.standards.contracts import clean

FINANCE_SHEET = "Table 1"
SHOUGANG_SHEET = "数据分类分级"


@dataclass(frozen=True)
class RawEntry:
    """One leaf row of a raw standard, with its traceable origin."""

    level_1: str = ""
    level_2: str = ""
    level_3: str = ""
    leaf: str = ""
    description: str = ""
    content: str = ""
    raw_level: str = ""
    resource: str = ""
    sheet: str = ""
    row: int | None = None

    def to_mapping(self) -> dict[str, Any]:
        return {
            "level_1": self.level_1,
            "level_2": self.level_2,
            "level_3": self.level_3,
            "leaf": self.leaf,
            "description": self.description,
            "content": self.content,
            "raw_level": self.raw_level,
            "resource": self.resource,
            "sheet": self.sheet,
            "row": self.row,
        }


@dataclass(frozen=True)
class ReaderResult:
    entries: tuple[RawEntry, ...]
    issues: tuple[str, ...] = ()


def _open_xlsx(path: Path, sheet: str):
    """Lazily import openpyxl so non-Excel code paths never require it."""
    try:
        import openpyxl
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("reading raw standard workbooks requires openpyxl") from exc
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    if sheet not in workbook.sheetnames:
        raise ValueError(f"sheet {sheet!r} not found in {path}")
    return workbook, workbook[sheet]


def read_finance_standard_guide(path: str | Path) -> ReaderResult:
    """Extract leaf rows of the finance grading-standard guide."""
    workbook, sheet = _open_xlsx(Path(path), FINANCE_SHEET)
    rows = list(sheet.iter_rows(values_only=True))
    workbook.close()

    entries: list[RawEntry] = []
    issues: list[str] = []
    level_1 = level_2 = level_3 = ""
    for index, row in enumerate(rows):
        if index < 2:  # two header rows
            continue
        if row[1] is not None:
            level_1 = clean(row[1])
        if row[2] is not None:
            level_2 = clean(row[2])
        if row[4] is not None:
            level_3 = clean(row[4])
        if row[6] is None:
            continue
        leaf = clean(row[6])
        if not leaf:
            issues.append(f"finance row {index + 1}: empty leaf")
            continue
        content = clean(row[7]) if row[7] is not None else ""
        raw_level = str(row[8]).strip() if row[8] is not None else ""
        entries.append(
            RawEntry(
                level_1=level_1,
                level_2=level_2,
                level_3=level_3,
                leaf=leaf,
                description=content,
                raw_level=raw_level,
                sheet=FINANCE_SHEET,
                row=index + 1,
            )
        )
    return ReaderResult(tuple(entries), tuple(issues))


@dataclass(frozen=True)
class _LevelBox:
    name: str = ""
    definition: str = ""


def read_guanji_catalog(path: str | Path) -> ReaderResult:
    """Extract leaf rows of the shougang (关基) grading catalog.

    A leaf is the DEEPEST level that carries a real name: the catalog encodes
    categories whose leaf sits at the 三级 level with a literal "——" in the
    四级 cell (e.g. 合同归并（B1-2）, 合同跟踪（B1-5）, 热轧作业计划（B3-3）).
    Those rows are real categories, NOT placeholders; their code/name come
    from the 三级 cell. The leaf's description is the definition column of
    its own level (四级 -> col 8; 三级 -> col 6). Literal cell text is kept
    lossless (no quality fixing).
    """
    workbook, sheet = _open_xlsx(Path(path), SHOUGANG_SHEET)
    rows = list(sheet.iter_rows(values_only=True))
    workbook.close()

    entries: list[RawEntry] = []
    issues: list[str] = []
    level_1 = _LevelBox()
    level_2 = _LevelBox()
    level_3 = _LevelBox()
    for index, row in enumerate(rows):
        if index < 2:  # title + header rows
            continue
        if len(row) < 11:
            issues.append(f"shougang row {index + 1}: too few columns")
            continue
        if row[1] is not None:
            level_1 = _LevelBox(clean(str(row[1])), clean(str(row[2])) if row[2] is not None else "")
        if row[3] is not None:
            level_2 = _LevelBox(clean(str(row[3])), clean(str(row[4])) if row[4] is not None else "")
        if row[5] is not None:
            level_3 = _LevelBox(clean(str(row[5])), clean(str(row[6])) if row[6] is not None else "")
        if row[7] is None:
            continue
        level_4 = _LevelBox(clean(str(row[7])), clean(str(row[8])) if row[8] is not None else "")
        # deepest real level is the leaf (non-empty, not the "——" marker)
        candidates = [
            (level_4.name, level_4.definition, "level_4"),
            (level_3.name, level_3.definition, "level_3"),
            (level_2.name, level_2.definition, "level_2"),
        ]
        leaf_name, leaf_definition, leaf_level = next(
            (c for c in candidates if c[0] and c[0] != "——"),
            ("", "", ""),
        )
        if not leaf_name:
            issues.append(f"shougang row {index + 1}: no real leaf level (all ——)")
            continue
        # keep only levels above/at the leaf; deeper levels are left empty
        resolved_level_3 = level_3.name if leaf_level in ("level_3", "level_4") else ""
        content = clean(str(row[9])) if row[9] is not None else ""
        raw_level = str(row[10]).strip() if row[10] is not None else ""
        resource = clean(str(row[11])) if len(row) > 11 and row[11] is not None else ""
        entries.append(
            RawEntry(
                level_1=level_1.name,
                level_2=level_2.name,
                level_3=resolved_level_3,
                leaf=leaf_name,
                description=leaf_definition,
                content=content,
                raw_level=raw_level,
                resource=resource,
                sheet=SHOUGANG_SHEET,
                row=index + 1,
            )
        )
    return ReaderResult(tuple(entries), tuple(issues))


__all__ = ["RawEntry", "ReaderResult", "read_finance_standard_guide", "read_guanji_catalog"]
