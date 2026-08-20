"""Phase 1 canonical standard — MergedCellResolver unit tests (hermetic).

Builds a tiny workbook on the fly, so these tests need openpyxl but no repo
data (skipped when openpyxl is missing).
"""

from __future__ import annotations

import pytest

openpyxl = pytest.importorskip("openpyxl")

from agent.standards.sources import MergedCellResolver  # noqa: E402


@pytest.fixture()
def wb(tmp_path):
    """A sheet with one vertical merge B2:B4 and one plain cell C2."""
    path = tmp_path / "t.xlsx"
    workbook = openpyxl.Workbook()
    ws = workbook.active
    ws.title = "S"
    ws["B2"] = "group-value"
    ws["C2"] = "plain-value"
    ws["C3"] = "plain-other"
    ws.merge_cells("B2:B4")
    workbook.save(path)
    workbook.close()
    return path


def test_anchor_cell(wb):
    resolver = MergedCellResolver(openpyxl.load_workbook(wb, data_only=True), "S")
    info = resolver.cell(2, "B")
    assert info.value == "group-value"
    assert info.anchor_cell == "B2"
    assert info.merged_range == "B2:B4"
    assert info.start_row == 2 and info.end_row == 4
    assert info.inherited is False
    assert info.to_mapping()["value"] == "group-value"


def test_inherited_cell(wb):
    resolver = MergedCellResolver(openpyxl.load_workbook(wb, data_only=True), "S")
    info = resolver.cell(4, "B")  # non-anchor inside the merge
    assert info.value == "group-value"  # anchor value expanded
    assert info.anchor_cell == "B2"
    assert info.merged_range == "B2:B4"
    assert info.inherited is True


def test_plain_cell(wb):
    resolver = MergedCellResolver(openpyxl.load_workbook(wb, data_only=True), "S")
    info = resolver.cell(2, "C")
    assert info.value == "plain-value"
    assert info.anchor_cell == "C2"
    assert info.merged_range is None
    assert info.inherited is False
    assert info.start_row is None


def test_column_letter_or_index_accepted(wb):
    resolver = MergedCellResolver(openpyxl.load_workbook(wb, data_only=True), "S")
    assert resolver.cell(2, "B").value == resolver.cell(2, 2).value == "group-value"
