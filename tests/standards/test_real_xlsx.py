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
        finance_raw.entries, source_file=str(FIN_XLSX), source_sheet="Table 1",
        reader_issues=finance_raw.issues,
    )
    shougang, shougang_report = build_shougang_standard(
        shougang_raw.entries, source_file=str(SHG_XLSX), source_sheet="数据分类分级",
        reader_issues=shougang_raw.issues,
    )
    return finance, finance_report, shougang, shougang_report


def test_finance_standard_is_lossless_237_entries_233_projection(built):
    finance, finance_report, _, _ = built
    registry = LeafRegistry.from_path(REG / "finance.registry.json")
    assert len(finance.entries) == 237  # one per real standard row — NO collapse
    assert finance.trainable_category_count() == 233
    assert finance_report.standard_entries_out == 237
    assert finance_report.training_categories == 233
    # training alias set == registry identity set (join stays compatible)
    assert {e.category_id for e in finance.entries} == set(registry.ids)
    # real hierarchy path depth preserved (dict compressed it)
    depths = {len(e.path) for e in finance.entries}
    assert 4 in depths and 3 in depths


def test_finance_five_level4_leaf_entries_kept_with_distinct_level3(built):
    finance, _, _, _ = built
    bucket = [
        e for e in finance.entries
        if e.category_id == "finance:业务.合约协议.基本信息"
    ]
    assert len(bucket) == 5  # 合同通用/贷款业务/中间业务/资金业务/其他支付业务
    assert len({e.standard_entry_id for e in bucket}) == 5
    # every entry keeps its own source row (nothing collapsed)
    assert len({e.source.row for e in bucket}) == 5


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


def test_finance_unresolved_evidence_lists_candidates(built):
    finance, _, _, _ = built
    records = load_canonical_records(CANON / "finance" / "all.json")
    report = align_dataset_to_standard(records, finance)
    assert report["sample_counts"]["unresolved"] == 37
    assert report["unresolved_by_status"] == {"missing_leaf": 34, "path_mismatch": 3}
    # evidence-only mapping exists and never repairs
    assert report["unresolved_evidence"]
    for item in report["unresolved_evidence"]:
        assert "candidate_standard_categories" in item


def test_shougang_standard_covers_registry_plus_lost_b3_6(built):
    _, _, shougang, _ = built
    registry = LeafRegistry.from_path(REG / "shougang.registry.json")
    standard_codes = {e.standard_entry_id for e in shougang.entries}
    assert len(shougang.entries) == 234
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
    _, _, _, _ = built
    # rebuild from the same raw readers — identical fingerprints
    f2, _ = build_finance_standard(
        read_finance_standard_guide(FIN_XLSX).entries,
        source_file=str(FIN_XLSX), source_sheet="Table 1",
    )
    s2, _ = build_shougang_standard(
        read_guanji_catalog(SHG_XLSX).entries,
        source_file=str(SHG_XLSX), source_sheet="数据分类分级",
    )
    assert len(f2.entries) == 237
    assert len(s2.entries) == 234
