"""Stage-3B tests: canonical dataset resolution against the LeafRegistry.

Fixture tests run on checked-in standards_map files / inline data (CI-safe).
Real-data integration assertions run only when data/<dataset>/all.json is
available locally and exercise the full build_canonical_dataset pipeline.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from agent.task import BUILTIN_DATASET_CONFIGS, ClassificationTargetResolver, LeafRegistry
from agent.task.canonical_corpus import (
    build_finance_corpus,
    build_shougang_corpus,
)
from agent.task.canonical_dataset import build_canonical_dataset, resolve_record
from agent.task.identity import code_leaf_map, leaf_registry_from_corpus
from agent.task.resolver import ResolutionStatus

ROOT = Path(__file__).resolve().parents[2]
STANDARDS = ROOT / "data" / "knowledge" / "standards_map"
GUANJI_PATH = STANDARDS / "guanji_dict.json"
FINANCIAL_PATH = STANDARDS / "financial_standards_dict.json"
REGISTRY_DIR = ROOT / "cfg" / "task" / "registry"
CORPUS_DIR = ROOT / "cfg" / "task" / "corpus"

PERS_LEAVES = [
    "人力资源数据", "任课信息", "基本信息年级信息和班级信息", "学历学位信息",
    "学校概况基本信息", "学生个人基本信息", "学生敏感个人信息", "学科类科研管理",
    "学籍基本数据", "学籍管理信息", "教职工个人基本信息", "毕业就业信息",
    "离退休信息", "系统数据", "考核信息", "考试成绩", "职称信息", "课程信息",
]


@pytest.fixture(autouse=True)
def _require_standards() -> None:
    if not GUANJI_PATH.is_file() or not FINANCIAL_PATH.is_file():
        pytest.skip("standards_map files not available")


@pytest.fixture(scope="module")
def finance_registry() -> LeafRegistry:
    categories, _ = build_finance_corpus(FINANCIAL_PATH, dataset="finance")
    return leaf_registry_from_corpus(categories)


@pytest.fixture(scope="module")
def guanji_registry() -> LeafRegistry:
    categories, _ = build_shougang_corpus(GUANJI_PATH, dataset="shougang")
    return leaf_registry_from_corpus(categories)


@pytest.fixture(scope="module")
def guanji_code_map() -> dict[str, str]:
    categories, _ = build_shougang_corpus(GUANJI_PATH, dataset="shougang")
    return code_leaf_map(categories)


@pytest.fixture(scope="module")
def pers_registry() -> LeafRegistry:
    from agent.task import CorpusCategory

    categories = [
        CorpusCategory(
            category_id=f"pers_info:{leaf}",
            name=leaf,
            path=(leaf,),
        )
        for leaf in PERS_LEAVES
    ]
    return leaf_registry_from_corpus(categories)


def _record(classification: dict[str, str], *, record_id: str = "r1") -> dict:
    return {
        "id": record_id,
        "key": "k",
        "label_status": "labeled",
        "classification": classification,
        "metadata": {"field_name": "f"},
        "data_level": "L2",
    }


# 1. resolved target.category_id is always in the registry ----------------------


def test_resolved_target_category_id_in_registry(
    finance_registry: LeafRegistry,
) -> None:
    resolver = ClassificationTargetResolver(BUILTIN_DATASET_CONFIGS["finance"])
    names = {c.name for c in finance_registry.categories}
    for record in [
        _record({"level_1": "客户", "level_2": "个人", "level_3": "", "level_4": "个人基本概况信息"}),
        _record({"level_1": "业务", "level_2": "交易信息", "level_3": "交易通用信息", "level_4": "交易基本信息"}),
    ]:
        result = resolve_record(record, resolver, finance_registry, names)
        assert result.status is ResolutionStatus.RESOLVED
        assert result.target is not None
        assert result.target.category_id in finance_registry.ids


# 2. finance level_3 does not affect the canonical ID ---------------------------


def test_finance_level3_not_part_of_identity(finance_registry: LeafRegistry) -> None:
    resolver = ClassificationTargetResolver(BUILTIN_DATASET_CONFIGS["finance"])
    names = {c.name for c in finance_registry.categories}
    # standard path for 配置信息 is 经营管理-技术管理-配置信息
    with_l3 = resolve_record(
        _record({"level_1": "经营管理", "level_2": "技术管理", "level_3": "系统管理信息", "level_4": "配置信息"}),
        resolver, finance_registry, names,
    )
    without_l3 = resolve_record(
        _record({"level_1": "经营管理", "level_2": "技术管理", "level_3": "", "level_4": "配置信息"}),
        resolver, finance_registry, names,
    )
    assert with_l3.status is ResolutionStatus.RESOLVED
    assert without_l3.status is ResolutionStatus.RESOLVED
    assert with_l3.target.category_id == without_l3.target.category_id
    assert with_l3.target.category_id == "finance:经营管理.技术管理.配置信息"
    # level_3 stays provenance in category_path
    assert with_l3.target.category_path == ("经营管理", "技术管理", "系统管理信息", "配置信息")


# 3. finance known path mismatch is never auto-fixed ----------------------------


def test_known_path_mismatch_not_auto_fixed(finance_registry: LeafRegistry) -> None:
    resolver = ClassificationTargetResolver(BUILTIN_DATASET_CONFIGS["finance"])
    names = {c.name for c in finance_registry.categories}
    # leaf 网络服务信息 exists in the standard under 经营管理/运营管理, but the
    # record carries 经营管理/技术管理 — must stay unresolved, no target
    result = resolve_record(
        _record({"level_1": "经营管理", "level_2": "技术管理", "level_3": "系统管理信息", "level_4": "网络服务信息"}),
        resolver, finance_registry, names,
    )
    assert result.status is ResolutionStatus.PATH_MISMATCH
    assert "网络服务信息" in names  # leaf is corpus-known
    # the canonical record must not carry a target (verified in the pipeline
    # test below); the informational target is only for audit details


# 4. finance missing leaf produces no target ------------------------------------


def test_missing_leaf_no_target(finance_registry: LeafRegistry) -> None:
    resolver = ClassificationTargetResolver(BUILTIN_DATASET_CONFIGS["finance"])
    names = {c.name for c in finance_registry.categories}
    result = resolve_record(
        _record({"level_1": "业务", "level_2": "账户信息", "level_3": "", "level_4": "交易清金额信息"}),
        resolver, finance_registry, names,
    )
    assert result.status is ResolutionStatus.MISSING_LEAF
    assert "交易清金额信息" not in names


# 5. shougang placeholder -------------------------------------------------------


def test_shougang_placeholder_is_placeholder(
    guanji_registry: LeafRegistry, guanji_code_map: dict[str, str],
) -> None:
    resolver = ClassificationTargetResolver(
        BUILTIN_DATASET_CONFIGS["shougang"], code_leaf_map=guanji_code_map
    )
    names = {c.name for c in guanji_registry.categories}
    result = resolve_record(
        _record({"level_1": "生产数据域", "level_2": "产出管理", "level_3": "实重管理", "level_4": "——"}),
        resolver, guanji_registry, names,
    )
    assert result.status is ResolutionStatus.PLACEHOLDER
    assert result.target is None
    assert "——" not in guanji_registry.ids  # placeholder never becomes a category


# 6. code strategy missing code -> code_unresolved ------------------------------


def test_code_unresolved_when_code_missing(guanji_registry: LeafRegistry) -> None:
    # code map deliberately incomplete (leaf '科研设备预约管理' has no entry)
    resolver = ClassificationTargetResolver(BUILTIN_DATASET_CONFIGS["shougang"], code_leaf_map={})
    names = {c.name for c in guanji_registry.categories}
    result = resolve_record(
        _record({"level_1": "研发数据域", "level_2": "科研管理", "level_3": "设备管理", "level_4": "科研设备预约管理"}),
        resolver, guanji_registry, names,
    )
    assert result.status is ResolutionStatus.CODE_UNRESOLVED
    assert result.target is None


# 7. infra resolves against the shared shougang registry ------------------------


def test_infra_four_leaves_resolve_to_shared_registry(
    guanji_registry: LeafRegistry, guanji_code_map: dict[str, str],
) -> None:
    config = BUILTIN_DATASET_CONFIGS["infra"]
    assert config.registry_source == "shougang"
    resolver = ClassificationTargetResolver(config, code_leaf_map=guanji_code_map)
    names = {c.name for c in guanji_registry.categories}
    leaves = ["厚板组板设计", "水质监测管理", "薄板合同产线专业化设计", "非常规委托检测管理"]
    for leaf in leaves:
        result = resolve_record(
            _record({"level_1": "生产数据域", "level_2": "生产合同（订单）", "level_3": "", "level_4": leaf}),
            resolver, guanji_registry, names,
        )
        assert result.status is ResolutionStatus.RESOLVED, leaf
        assert result.target.category_id in guanji_registry.ids


# 8. pers_info without description still resolves -------------------------------


def test_pers_info_without_description_still_resolves(pers_registry: LeafRegistry) -> None:
    resolver = ClassificationTargetResolver(BUILTIN_DATASET_CONFIGS["pers_info"])
    names = {c.name for c in pers_registry.categories}
    for leaf in PERS_LEAVES:
        result = resolve_record(
            _record({"level_1": "", "level_2": "", "level_3": "", "level_4": leaf}),
            resolver, pers_registry, names,
        )
        assert result.status is ResolutionStatus.RESOLVED, leaf
        assert result.target.category_id == f"pers_info:{leaf}"


# 9/10. full pipeline: classification preserved, unresolved not dropped ----------


def _write_tmp_dataset(tmp_path: Path, dataset: str, records: list[dict]) -> None:
    data_dir = tmp_path / "data"
    out = data_dir / dataset / "all.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False)


def test_pipeline_preserves_classification_and_keeps_unresolved(
    tmp_path: Path, finance_registry: LeafRegistry,
) -> None:
    records = [
        _record(
            {"level_1": "经营 管理", "level_2": "技术管理", "level_3": "系统管理信息", "level_4": "配置信息"},
            record_id="resolved-1",
        ),
        _record(
            {"level_1": "经营管理", "level_2": "技术管理", "level_3": "系统管理信息", "level_4": "网络服务信息"},
            record_id="mismatch-1",
        ),
        _record(
            {"level_1": "业务", "level_2": "账户信息", "level_3": "", "level_4": "交易清金额信息"},
            record_id="missing-1",
        ),
    ]
    _write_tmp_dataset(tmp_path, "finance", records)
    result = build_canonical_dataset(
        "finance",
        data_dir=tmp_path / "data",
        registry_dir=REGISTRY_DIR,
        corpus_dir=CORPUS_DIR,
        overwrite=False,
    )
    assert result.input_records == 3
    assert result.output_records == 3
    assert result.status_counts == {
        "missing_leaf": 1,
        "path_mismatch": 1,
        "resolved": 1,
    }
    assert result.resolved_targets_in_registry is True

    with (tmp_path / "data" / "finance" / "canonical" / "all.json").open(encoding="utf-8") as handle:
        canonical = json.load(handle)
    by_id = {record["id"]: record for record in canonical}
    # classification is provenance, never rewritten (whitespace preserved)
    assert by_id["resolved-1"]["classification"]["level_1"] == "经营 管理"
    assert by_id["resolved-1"]["resolution_status"] == "resolved"
    target = by_id["resolved-1"]["target"]
    assert target["category_id"] == "finance:经营管理.技术管理.配置信息"
    assert target["category_path"] == ["经营 管理", "技术管理", "系统管理信息", "配置信息"]
    # unresolved records are kept, tagged, and traceable by original id
    assert by_id["mismatch-1"]["resolution_status"] == "path_mismatch"
    assert "target" not in by_id["mismatch-1"]
    assert by_id["missing-1"]["resolution_status"] == "missing_leaf"
    assert "target" not in by_id["missing-1"]
    assert result.unresolved_details["record_ids"] == ["mismatch-1", "missing-1"]


# 11/12/13. CLI integration: fresh out-dir, fail-fast, deterministic -------------


def _run_cli(tmp_path: Path, *extra: str) -> int:
    from script.canonical import targets as targets_cli

    return targets_cli.main(
        [
            "--data-dir", str(tmp_path / "data"),
            "--registry-dir", str(REGISTRY_DIR),
            "--corpus-dir", str(CORPUS_DIR),
            *extra,
        ]
    )


def test_cli_first_run_succeeds_and_second_fails(tmp_path: Path) -> None:
    records = [
        _record({"level_1": "客户", "level_2": "个人", "level_3": "", "level_4": "个人基本概况信息"}),
        _record({"level_1": "业务", "level_2": "交易信息", "level_3": "交易通用信息", "level_4": "交易基本信息"}),
        _record({"level_1": "业务", "level_2": "账户信息", "level_3": "", "level_4": "交易清金额信息"}),
    ]
    _write_tmp_dataset(tmp_path, "finance", records)
    # first run without --overwrite on a fresh out-dir succeeds
    assert _run_cli(tmp_path, "--dataset", "finance") == 0
    assert (tmp_path / "data" / "finance" / "canonical" / "all.json").is_file()
    assert (tmp_path / "data" / "finance" / "canonical" / "resolution_report.json").is_file()
    # second run without --overwrite fails fast before writing anything
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _run_cli(tmp_path, "--dataset", "finance")


def test_cli_is_deterministic_across_two_runs(tmp_path: Path) -> None:
    records = [
        _record({"level_1": "客户", "level_2": "个人", "level_3": "", "level_4": "个人基本概况信息"}),
        _record({"level_1": "经营管理", "level_2": "技术管理", "level_3": "系统管理信息", "level_4": "网络服务信息"}),
    ]
    _write_tmp_dataset(tmp_path, "finance", records)
    assert _run_cli(tmp_path, "--dataset", "finance") == 0
    first_all = (tmp_path / "data" / "finance" / "canonical" / "all.json").read_bytes()
    first_report = (tmp_path / "data" / "finance" / "canonical" / "resolution_report.json").read_bytes()
    assert _run_cli(tmp_path, "--dataset", "finance", "--overwrite") == 0
    second_all = (tmp_path / "data" / "finance" / "canonical" / "all.json").read_bytes()
    second_report = (tmp_path / "data" / "finance" / "canonical" / "resolution_report.json").read_bytes()
    assert first_all == second_all
    assert first_report == second_report


# --- edge cases: code_unresolved pipeline, cross-dataset fail-fast, invalid ----


def _write_tmp_corpus(tmp_path: Path, dataset: str, categories: list[dict]) -> None:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    with (corpus_dir / f"{dataset}.corpus.json").open("w", encoding="utf-8") as handle:
        json.dump({"dataset": dataset, "categories": categories, "build_report": {}}, handle, ensure_ascii=False)


def test_code_unresolved_pipeline_no_crash(tmp_path: Path) -> None:
    """code strategy + missing code: status=code_unresolved, no target in the
    canonical record, report.by_leaf correct, pipeline must not crash."""
    _write_tmp_dataset(
        tmp_path,
        "shougang",
        [
            _record(
                {"level_1": "研发数据域", "level_2": "科研管理", "level_3": "设备管理", "level_4": "科研设备预约管理"},
                record_id="has-code",
            ),
            _record(
                {"level_1": "研发数据域", "level_2": "科研管理", "level_3": "设备管理", "level_4": "科研进程管控"},
                record_id="no-code",
            ),
        ],
    )
    # incomplete canonical corpus: 科研设备预约管理 has a code, 科研进程管控 has none
    _write_tmp_corpus(
        tmp_path,
        "shougang",
        [
            {"category_id": "A1-1-1", "name": "科研设备预约管理", "code": "A1-1-1"},
        ],
    )
    result = build_canonical_dataset(
        "shougang",
        data_dir=tmp_path / "data",
        registry_dir=REGISTRY_DIR,  # real shared guanji registry (233)
        corpus_dir=tmp_path / "corpus",
    )
    assert result.status_counts == {"code_unresolved": 1, "resolved": 1}
    assert result.unresolved_details["code_unresolved"] == {
        "count": 1,
        "by_leaf": {"科研进程管控": 1},
    }
    with (tmp_path / "data" / "shougang" / "canonical" / "all.json").open(encoding="utf-8") as handle:
        canonical = json.load(handle)
    by_id = {record["id"]: record for record in canonical}
    assert by_id["has-code"]["resolution_status"] == "resolved"
    assert by_id["has-code"]["target"]["category_id"] == "A1-1-1"
    assert by_id["no-code"]["resolution_status"] == "code_unresolved"
    assert "target" not in by_id["no-code"]
    assert result.unresolved_details["record_ids"] == ["no-code"]


def test_cross_dataset_fail_fast_no_partial_writes(tmp_path: Path) -> None:
    """dataset A valid, dataset B missing input: --datasets A B must fail and
    neither A nor B may produce canonical output."""
    from script.canonical import targets as targets_cli

    _write_tmp_dataset(
        tmp_path,
        "finance",
        [_record({"level_1": "客户", "level_2": "个人", "level_3": "", "level_4": "个人基本概况信息"})],
    )
    # shougang input deliberately missing
    with pytest.raises(FileNotFoundError, match="input dataset not found"):
        targets_cli.main(
            [
                "--data-dir", str(tmp_path / "data"),
                "--registry-dir", str(REGISTRY_DIR),
                "--corpus-dir", str(CORPUS_DIR),
                "--datasets", "finance", "shougang",
            ]
        )
    assert not (tmp_path / "data" / "finance" / "canonical").exists()
    assert not (tmp_path / "data" / "shougang" / "canonical").exists()


def test_invalid_record_non_mapping_is_audited(tmp_path: Path) -> None:
    """Non-object records (bare strings in the JSON array) resolve to
    invalid_record instead of raising AttributeError, and stay auditable."""
    data_dir = tmp_path / "data"
    out = data_dir / "finance" / "all.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(
            [
                _record({"level_1": "客户", "level_2": "个人", "level_3": "", "level_4": "个人基本概况信息"}, record_id="ok"),
                "not-a-record",
                42,
                None,
            ],
            handle,
            ensure_ascii=False,
        )
    result = build_canonical_dataset(
        "finance",
        data_dir=data_dir,
        registry_dir=REGISTRY_DIR,
        corpus_dir=CORPUS_DIR,
    )
    assert result.status_counts == {"invalid_record": 3, "resolved": 1}
    with (data_dir / "finance" / "canonical" / "all.json").open(encoding="utf-8") as handle:
        canonical = json.load(handle)
    invalid = [r for r in canonical if r["resolution_status"] == "invalid_record"]
    assert len(invalid) == 3
    assert all("target" not in r for r in invalid)
    assert invalid[0]["record"] == "not-a-record"  # original value kept, auditable
    assert invalid[2]["record"] is None
    assert result.unresolved_details["record_ids"] == ["", "", ""]


# ---------------------------------------------------------------------------
# real-data integration assertions (local only; skipped in CI without data/)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_data_dir() -> Path | None:
    path = ROOT / "data"
    if not (path / "finance" / "all.json").is_file():
        return None
    return path


def _copy_real_input(tmp_path: Path, dataset: str, real_data: Path) -> Path:
    src = real_data / dataset / "all.json"
    dst = tmp_path / "data" / dataset / "all.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    return dst


def test_real_finance_counts(tmp_path: Path, real_data_dir: Path | None) -> None:
    if real_data_dir is None:
        pytest.skip("real data/ not available")
    _copy_real_input(tmp_path, "finance", real_data_dir)
    result = build_canonical_dataset(
        "finance", data_dir=tmp_path / "data",
        registry_dir=REGISTRY_DIR, corpus_dir=CORPUS_DIR,
    )
    assert result.input_records == 568
    assert result.status_counts == {
        "resolved": 531,
        "missing_leaf": 34,
        "path_mismatch": 3,
    }
    assert result.resolved_targets_in_registry is True
    assert result.unresolved_details["missing_leaf"]["by_leaf"] == {
        "交易清金额信息": 1,
        "单位基本信息": 3,
        "单位基本情况": 19,
        "单位联系人信息": 9,
        "基本信息（公开": 2,
    }
    assert result.unresolved_details["path_mismatch"]["by_id"] == {
        "finance:业务.账户信息.交易清结算信息": 1,
        "finance:经营管理.技术管理.网络服务信息": 2,
    }


def test_real_infra_counts(tmp_path: Path, real_data_dir: Path | None) -> None:
    if real_data_dir is None:
        pytest.skip("real data/ not available")
    _copy_real_input(tmp_path, "infra", real_data_dir)
    result = build_canonical_dataset(
        "infra", data_dir=tmp_path / "data",
        registry_dir=REGISTRY_DIR, corpus_dir=CORPUS_DIR,
    )
    assert result.status_counts == {"resolved": 64}
    assert result.registry_size == 233  # shared shougang universe


def test_real_pers_info_counts(tmp_path: Path, real_data_dir: Path | None) -> None:
    if real_data_dir is None:
        pytest.skip("real data/ not available")
    _copy_real_input(tmp_path, "pers_info", real_data_dir)
    result = build_canonical_dataset(
        "pers_info", data_dir=tmp_path / "data",
        registry_dir=REGISTRY_DIR, corpus_dir=CORPUS_DIR,
    )
    assert result.status_counts == {"resolved": 176}
    assert result.registry_size == 18


def test_real_shougang_counts(tmp_path: Path, real_data_dir: Path | None) -> None:
    if real_data_dir is None:
        pytest.skip("real data/ not available")
    _copy_real_input(tmp_path, "shougang", real_data_dir)
    result = build_canonical_dataset(
        "shougang", data_dir=tmp_path / "data",
        registry_dir=REGISTRY_DIR, corpus_dir=CORPUS_DIR,
    )
    assert result.status_counts == {
        "resolved": 18393,
        "placeholder": 1022,
    }
    assert result.resolved_targets_in_registry is True
    assert result.unresolved_details["missing_leaf"]["count"] == 0
    assert result.unresolved_details["code_unresolved"]["count"] == 0
    assert result.unresolved_details["path_mismatch"]["count"] == 0
    # every unresolved record stays traceable (placeholder records included)
    assert len(result.unresolved_details["record_ids"]) == 1022
