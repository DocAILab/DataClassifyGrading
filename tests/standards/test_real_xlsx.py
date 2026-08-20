"""Phase 1 canonical standard — integration tests against real raw workbooks.

These read the ORIGINAL standard workbooks under data/raw (gitignored) and the
canonical dataset layer. They are skipped when the raw files are absent (CI /
fresh clone) so the suite stays green without the data provider's files.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.standards.align import align_dataset_to_standard, load_canonical_records
from agent.standards.build import (
    build_finance_standard,
    build_shougang_standard,
    resolve_standard_dataset,
)
from agent.standards.sources import (
    read_finance_standard_guide,
    read_guanji_catalog,
)
from agent.task import LeafRegistry

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
FIN_XLSX = RAW / "金融行业数据安全分类分级标准指南.xlsx"
SHG_XLSX = RAW / "关基-数据分类分级目录.xlsx"
CANON = ROOT / "data" / "canonical"
REG = ROOT / "cfg" / "task" / "registry"

pytestmark = pytest.mark.skipif(
    not (FIN_XLSX.is_file() and SHG_XLSX.is_file()),
    reason="raw standard workbooks not present (data/raw is gitignored)",
)


@pytest.fixture(scope="module")
def built():
    finance_raw = read_finance_standard_guide(FIN_XLSX)
    shougang_raw = read_guanji_catalog(SHG_XLSX)
    finance, finance_report = build_finance_standard(
        finance_raw.entries, source_file=str(FIN_XLSX), source_sheet="Table 1"
    )
    shougang, shougang_report = build_shougang_standard(
        shougang_raw.entries, source_file=str(SHG_XLSX), source_sheet="数据分类分级"
    )
    return finance, finance_report, shougang, shougang_report


def test_finance_standard_matches_registry_identity(built):
    finance, _, _, _ = built
    registry = LeafRegistry.from_path(REG / "finance.registry.json")
    assert len(finance.categories) == 233
    assert {c.category_id for c in finance.categories} == set(registry.ids)
    # real hierarchy path depth is preserved (E2E: dict had compressed L1-L2-leaf)
    depths = {len(c.path) for c in finance.categories}
    assert 4 in depths and 3 in depths


def test_finance_unparseable_levels_reported_not_fixed(built):
    _, finance_report, _, _ = built
    unparseable = [i for i in finance_report.issues if i.kind == "level_unparseable"]
    assert len(unparseable) == 2
    assert all("standard_data_level=null (not guessed)" in i.detail for i in unparseable)


def test_finance_alignment_reproduces_known_outliers(built):
    finance, _, _, _ = built
    records = load_canonical_records(CANON / "finance" / "all.json")
    report = align_dataset_to_standard(records, finance)
    counts = report["sample_counts"]
    assert counts["total"] == 568
    assert counts["resolved"] == 531
    assert counts["matched"] == 529
    assert counts["mismatched"] == 2
    assert counts["standard_missing"] == 0
    fields = {m["field"] for m in report["mismatched_samples"]}
    assert fields == {"AMONEY", "HXTRADENO"}


def test_shougang_standard_covers_registry_plus_lost_b3_6(built):
    _, _, shougang, _ = built
    registry = LeafRegistry.from_path(REG / "shougang.registry.json")
    standard_codes = {c.category_id for c in shougang.categories}
    assert len(shougang.categories) == 234
    assert set(registry.ids) <= standard_codes
    assert standard_codes - set(registry.ids) == {"B3-6"}  # 中厚板作业计划 lost by legacy


def test_shougang_alignment_100_percent(built):
    _, _, shougang, _ = built
    records = load_canonical_records(CANON / "shougang" / "all.json")
    report = align_dataset_to_standard(records, shougang)
    counts = report["sample_counts"]
    assert counts["total"] == 19415
    assert counts["resolved"] == 18393
    assert counts["matched"] == 18393
    assert counts["mismatched"] == 0
    assert counts["standard_missing"] == 0


def test_infra_reuses_shougang_standard(built):
    _, _, shougang, _ = built
    assert resolve_standard_dataset("infra") == "shougang"
    records = load_canonical_records(CANON / "infra" / "all.json")
    report = align_dataset_to_standard(records, shougang)
    counts = report["sample_counts"]
    assert counts["total"] == 64
    assert counts["matched"] == 64
    assert counts["mismatched"] == 0


def test_pers_info_has_no_canonical_standard():
    assert resolve_standard_dataset("pers_info") is None


def test_build_deterministic_on_real_inputs(built):
    finance, _, shougang, _ = built
    # rebuild from the same raw readers — identical fingerprints
    f2, _ = build_finance_standard(
        read_finance_standard_guide(FIN_XLSX).entries,
        source_file=str(FIN_XLSX), source_sheet="Table 1",
    )
    s2, _ = build_shougang_standard(
        read_guanji_catalog(SHG_XLSX).entries,
        source_file=str(SHG_XLSX), source_sheet="数据分类分级",
    )
    assert finance.fingerprint() == f2.fingerprint()
    assert shougang.fingerprint() == s2.fingerprint()
