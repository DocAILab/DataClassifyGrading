"""RL sample + VeRL parquet exporter/validator contract (stage 4A)."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from agent.task import LeafRegistry, TaskConfig
from agent.task.canonical_dataset import load_corpus_categories
from agent.training.rl import (
    VERL_RL_COLUMNS,
    build_rl_row,
    build_rl_samples,
    export_rl_dataset,
    validate_rl_dataset,
)

FIXTURE = Path(__file__).resolve().parents[1] / "sft" / "fixtures"
METADATA_FIELDS = ["field_name", "field_description"]


@pytest.fixture(scope="module")
def registry() -> LeafRegistry:
    return LeafRegistry.from_path(FIXTURE / "registry.json")


@pytest.fixture(scope="module")
def corpus():
    return {
        category.category_id: category
        for category in load_corpus_categories(FIXTURE / "corpus.json")
    }


@pytest.fixture(scope="module")
def config() -> TaskConfig:
    return TaskConfig.from_mapping({"metadata_fields": list(METADATA_FIELDS)})


def _canonical_with_edge_records(tmp_path: Path) -> Path:
    """Fixture canonical + 5 records: resolved, resolved with mismatched
    classification level_4 (provenance only), unresolved (must not enter),
    plus one resolved record per val/test split."""
    with (FIXTURE / "canonical" / "all.json").open(encoding="utf-8") as handle:
        records = json.load(handle)
    base = next(record for record in records if record["id"] == "fixture-1")
    mismatched = copy.deepcopy(base)
    mismatched["id"] = "fixture-mismatch"
    mismatched["classification"]["level_4"] = "D"  # contradicts target.category_id "C"
    unresolved = copy.deepcopy(base)
    unresolved["id"] = "fixture-unresolved"
    unresolved["resolution_status"] = "missing_leaf"
    unresolved.pop("target", None)
    val_record = copy.deepcopy(base)
    val_record["id"] = "fixture-val-1"
    test_record = copy.deepcopy(base)
    test_record["id"] = "fixture-test-1"
    path = tmp_path / "canonical" / "all.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            [base, mismatched, unresolved, val_record, test_record], ensure_ascii=False
        ),
        encoding="utf-8",
    )
    return path


def _split_dir(tmp_path: Path) -> Path:
    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    (split_dir / "train.json").write_text(
        json.dumps([{"id": "fixture-1"}, {"id": "fixture-mismatch"}]),
        encoding="utf-8",
    )
    (split_dir / "val.json").write_text(
        json.dumps([{"id": "fixture-val-1"}, {"id": "fixture-unresolved"}]),
        encoding="utf-8",
    )
    (split_dir / "test.json").write_text(
        json.dumps([{"id": "fixture-test-1"}]), encoding="utf-8"
    )
    return split_dir


@pytest.fixture(scope="module")
def exported(tmp_path_factory, registry, corpus, config):
    tmp = tmp_path_factory.mktemp("rl-dataset")
    canonical = _canonical_with_edge_records(tmp)
    split_dir = _split_dir(tmp)
    out = tmp / "out"
    report = export_rl_dataset(
        canonical,
        split_dir,
        out,
        "fixture_ds",
        registry,
        config,
        corpus=corpus,
    )
    return {"out": out, "report": report, "canonical": canonical, "split_dir": split_dir}


def test_unresolved_records_do_not_enter(exported) -> None:
    report = exported["report"]
    assert report["trainable_resolved"] == 4  # all resolved records are inside some split
    # val keeps one resolved record but must skip the unresolved one
    assert report["splits"]["val"]["exported_rows"] == 2  # fixture-val-1 x 2 stages
    assert report["splits"]["val"]["skipped_not_resolved"] == 1
    assert report["splits"]["train"]["exported_rows"] == 4  # 2 records x 2 stages
    assert report["splits"]["test"]["exported_rows"] == 2


def test_target_category_id_is_the_only_label(exported) -> None:
    rows = pq.read_table(exported["out"] / "train.parquet").to_pylist()
    by_source = {row["extra_info"]["source_id"]: row for row in rows}
    mismatch = by_source["fixture-mismatch"]
    # classification.level_4 is "D" but the canonical target is "C"
    assert mismatch["reward_model"]["ground_truth"] == "C"


def test_stage1_prompt_uses_full_registry(exported, registry) -> None:
    from agent.task import PromptChoiceRegistry

    rows = pq.read_table(exported["out"] / "train.parquet").to_pylist()
    stage1 = next(row for row in rows if row["extra_info"]["stage"] == "stage1")
    user = stage1["prompt"][1]["content"]
    # compact [choice_id, display_name] catalog; canonical ids never shown
    assert '"category_id"' not in user
    catalog = json.loads(user.split("\n", 1)[1].split("\nField metadata:", 1)[0])
    assert len(catalog) == len(registry.categories)
    assert [entry[0] for entry in catalog] == [
        str(index) for index in range(1, len(registry.categories) + 1)
    ]
    assert len({entry[1] for entry in catalog}) == len(registry.categories)
    choices = PromptChoiceRegistry.from_registry(registry)
    assert [choices.category_id_of(entry[0]) for entry in catalog] == list(registry.ids)


def test_stage2_corpus_lookup_by_category_id(exported, corpus) -> None:
    rows = pq.read_table(exported["out"] / "train.parquet").to_pylist()
    stage2 = next(row for row in rows if row["extra_info"]["stage"] == "stage2")
    candidates = stage2["extra_info"]["candidates"]
    assert len(candidates) == 5 and len(set(candidates)) == 5
    for candidate in candidates:
        assert candidate in corpus, candidate
    # bundle content comes from the corpus, not just registry names
    assert '"descriptions"' in stage2["prompt"][1]["content"] or "examples" in stage2["prompt"][1]["content"]


def test_ground_truth_is_canonical_target_not_classification(exported, registry) -> None:
    rows = pq.read_table(exported["out"] / "train.parquet").to_pylist()
    for row in rows:
        gt = row["reward_model"]["ground_truth"]
        assert gt in registry.ids
        source = row["extra_info"]["source_id"]
        if source == "fixture-mismatch":
            assert gt == "C"  # never "D" from classification.level_4


def test_parquet_columns_are_exactly_the_verl_five(exported) -> None:
    for split in ("train", "test"):
        table = pq.read_table(exported["out"] / f"{split}.parquet")
        assert set(table.column_names) == set(VERL_RL_COLUMNS)


def test_prompt_has_no_assistant_gold(exported) -> None:
    rows = pq.read_table(exported["out"] / "train.parquet").to_pylist()
    for row in rows:
        roles = [message["role"] for message in row["prompt"]]
        assert roles == ["system", "user"]
        assert not any(
            token in json.dumps(row["prompt"], ensure_ascii=False)
            for token in ("<|im_start|>", "<|im_end|>")
        )


def test_reward_model_and_extra_info_contract(exported, config) -> None:
    rows = pq.read_table(exported["out"] / "train.parquet").to_pylist()
    for row in rows:
        assert row["data_source"] in {"fixture_ds/stage1", "fixture_ds/stage2"}
        assert row["reward_model"] == {
            "style": "rule",
            "ground_truth": row["reward_model"]["ground_truth"],
        }
        extra = row["extra_info"]
        assert extra["dataset"] == "fixture_ds"
        assert extra["stage"] in {"stage1", "stage2"}
        assert f"{extra['dataset']}/{extra['stage']}" == row["data_source"]
        assert isinstance(extra["source_id"], str) and extra["source_id"]
        assert set(extra["metadata"]) == set(config.metadata_fields)
        assert "candidates" not in extra or extra["candidates"] is None or extra["stage"] == "stage2"
    # no training-algorithm internal fields
    for row in rows:
        assert not any(
            key in row for key in ("advantage", "old_logp", "rollout", "policy", "value")
        )


def test_export_is_deterministic(exported, tmp_path, registry, corpus, config) -> None:
    out2 = tmp_path / "out-again"
    export_rl_dataset(
        exported["canonical"],
        exported["split_dir"],
        out2,
        "fixture_ds",
        registry,
        config,
        corpus=corpus,
    )
    for split in ("train", "val", "test"):
        assert (exported["out"] / f"{split}.parquet").read_bytes() == (
            out2 / f"{split}.parquet"
        ).read_bytes()


def test_validate_accepts_real_exports(exported, registry, corpus, config) -> None:
    report = validate_rl_dataset(
        exported["out"], "fixture_ds", registry, config, corpus=corpus
    )
    assert report["valid"] is True, report


def test_validate_reports_corrupted_rows(tmp_path, registry, corpus, config) -> None:
    out = tmp_path / "corrupted"
    export_rl_dataset(
        _canonical_with_edge_records(tmp_path),
        _split_dir(tmp_path),
        out,
        "fixture_ds",
        registry,
        config,
        corpus=corpus,
    )
    import pyarrow as pa

    rows = pq.read_table(out / "train.parquet").to_pylist()
    rows[0]["reward_model"]["ground_truth"] = "NOT_IN_REGISTRY"
    pq.write_table(pa.Table.from_pylist(rows), out / "train.parquet")
    report = validate_rl_dataset(out, "fixture_ds", registry, config, corpus=corpus)
    assert report["valid"] is False
    assert any("ground_truth" in error for error in report["splits"]["train"]["errors"])


def test_fully_unresolved_split_fails_fast(tmp_path, registry, corpus, config) -> None:
    canonical = _canonical_with_edge_records(tmp_path)
    split_dir = tmp_path / "splits-empty"
    split_dir.mkdir()
    (split_dir / "train.json").write_text(
        json.dumps([{"id": "fixture-unresolved"}]), encoding="utf-8"
    )
    (split_dir / "val.json").write_text(
        json.dumps([{"id": "fixture-1"}]), encoding="utf-8"
    )
    (split_dir / "test.json").write_text(
        json.dumps([{"id": "fixture-1"}]), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="no resolved records"):
        export_rl_dataset(
            canonical, split_dir, tmp_path / "out-empty", "fixture_ds", registry, config, corpus=corpus
        )


def _canonical_with_outside_bad_target(tmp_path: Path) -> Path:
    """Canonical edge records + one resolved record NOT in any split whose
    target.category_id is absent from the registry."""
    edge = _canonical_with_edge_records(tmp_path)
    with edge.open(encoding="utf-8") as handle:
        records = json.load(handle)
    with (FIXTURE / "canonical" / "all.json").open(encoding="utf-8") as handle:
        base = next(
            record for record in json.load(handle) if record["id"] == "fixture-1"
        )
    bad = copy.deepcopy(base)
    bad["id"] = "fixture-outside-bad"
    bad["target"] = copy.deepcopy(bad["target"])
    bad["target"]["category_id"] = "NOT_IN_REGISTRY"
    records.append(bad)
    path = tmp_path / "canonical-bad" / "all.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    return path


def test_resolved_outside_split_with_bad_target_fails_fast(tmp_path, registry, corpus, config) -> None:
    """Contract validation runs at index build, independent of split membership:
    a resolved record absent from every split but with target.category_id outside
    the registry must fail the export."""
    canonical = _canonical_with_outside_bad_target(tmp_path)
    with pytest.raises(ValueError, match="absent from the leaf registry"):
        export_rl_dataset(
            canonical,
            _split_dir(tmp_path),
            tmp_path / "out-bad",
            "fixture_ds",
            registry,
            config,
            corpus=corpus,
        )


def _export_valid(tmp_path: Path, name: str, registry, corpus, config) -> Path:
    out = tmp_path / name
    export_rl_dataset(
        _canonical_with_edge_records(tmp_path),
        _split_dir(tmp_path),
        out,
        "fixture_ds",
        registry,
        config,
        corpus=corpus,
    )
    return out


def _rewrite_split(path: Path, mutator) -> None:
    import pyarrow as pa

    rows = pq.read_table(path).to_pylist()
    mutator(rows)
    pq.write_table(pa.Table.from_pylist(rows), path)


DUPLICATE_OR_MISSING_CASES = [
    # (label, mutator, expected substring in the stage-pair error)
    (
        "duplicate-stage1",
        lambda rows: rows.append(
            copy.deepcopy(
                next(
                    r for r in rows if r["extra_info"]["stage"] == "stage1"
                )
            )
        ),
        "found 2 stage1, 1 stage2",
    ),
    (
        "duplicate-stage2",
        lambda rows: rows.append(
            copy.deepcopy(
                next(
                    r for r in rows if r["extra_info"]["stage"] == "stage2"
                )
            )
        ),
        "found 1 stage1, 2 stage2",
    ),
    (
        "missing-stage2",
        lambda rows: rows.__setitem__(
            slice(None),
            [r for r in rows if r["extra_info"]["stage"] != "stage2"],
        ),
        "found 1 stage1, 0 stage2",
    ),
    (
        "missing-stage1",
        lambda rows: rows.__setitem__(
            slice(None),
            [r for r in rows if r["extra_info"]["stage"] != "stage1"],
        ),
        "found 0 stage1, 1 stage2",
    ),
]


@pytest.mark.parametrize(
    ("label", "mutator", "expected"),
    DUPLICATE_OR_MISSING_CASES,
)
def test_validate_rejects_duplicate_or_missing_stage_rows(
    label, mutator, expected, tmp_path, registry, corpus, config
) -> None:
    """Every source_id must have exactly one stage1 row and one stage2 row;
    duplicate stage rows and missing stages are validator failures."""
    out = _export_valid(tmp_path, f"out-{label}", registry, corpus, config)
    _rewrite_split(out / "train.parquet", mutator)
    report = validate_rl_dataset(out, "fixture_ds", registry, config, corpus=corpus)
    assert report["valid"] is False
    assert any(
        "exactly one stage1 row and one stage2 row" in error and expected in error
        for error in report["splits"]["train"]["errors"]
    ), report["splits"]["train"]["errors"]


def _six_registry() -> LeafRegistry:
    base = LeafRegistry.from_path(FIXTURE / "registry.json")
    from agent.task import LeafCategory

    return LeafRegistry(
        tuple(base.categories) + (LeafCategory(category_id="F", name="zeta data", description="zeta data"),)
    )


def _six_corpus():
    from agent.task import CorpusCategory

    base = {
        category.category_id: category
        for category in load_corpus_categories(FIXTURE / "corpus.json")
    }
    base["F"] = CorpusCategory(
        category_id="F", name="zeta data", description="zeta data"
    )
    return base


def test_ground_truth_not_required_in_stage2_candidates(
    tmp_path, config, monkeypatch
) -> None:
    """Two-stage semantics: a Stage 1 recall failure (ground_truth not in the
    candidate bundle) is legal, not a Stage 2 contract error. Stage 2 rows stay
    valid, the parser runs normally, and a valid candidate answer still computes
    a reward without raising."""
    import agent.training.rl.sample as rl_sample_module

    def candidates_excluding_gt(ground_truth: str, reg: LeafRegistry):
        return [candidate for candidate in reg.ids if candidate != ground_truth][:5]

    monkeypatch.setattr(
        rl_sample_module, "build_candidates", candidates_excluding_gt
    )
    registry = _six_registry()
    corpus = _six_corpus()
    out = _export_valid(tmp_path, "out-gt-out", registry, corpus, config)

    rows = pq.read_table(out / "train.parquet").to_pylist()
    stage2 = next(row for row in rows if row["extra_info"]["stage"] == "stage2")
    ground_truth = stage2["reward_model"]["ground_truth"]
    candidates = stage2["extra_info"]["candidates"]
    assert ground_truth not in candidates

    # validator accepts: GT missing from candidates is NOT a contract error
    report = validate_rl_dataset(out, "fixture_ds", registry, config, corpus=corpus)
    assert report["valid"] is True, report

    # parser runs normally on a valid candidate answer
    from agent.task import check_stage2_output

    answer = candidates[0]
    parsed = check_stage2_output(
        json.dumps({"answer": answer}, ensure_ascii=False), candidates=candidates
    )
    assert parsed.ok is True

    # reward computes a valid candidate answer without raising (not the GT -> partial)
    from agent.training.rl import RewardResult, reward_stage2

    reward = reward_stage2(
        json.dumps({"answer": answer}, ensure_ascii=False),
        ground_truth=ground_truth,
        candidates=candidates,
        registry=registry,
    )
    assert isinstance(reward, RewardResult)
    assert reward.format_valid is True
    assert reward.task_correct is False
    assert reward.reward < 1.0


def test_build_rl_samples_requires_resolved(registry, corpus, config) -> None:
    with (FIXTURE / "canonical" / "all.json").open(encoding="utf-8") as handle:
        record = json.load(handle)[0]
    stage1, stage2 = build_rl_samples(
        record, 0, FIXTURE / "canonical" / "all.json",
        dataset="fixture_ds", registry=registry, config=config, corpus=corpus,
    )
    assert stage1.stage == "stage1" and stage1.candidates is None
    assert stage2.stage == "stage2" and stage2.candidates is not None
    assert stage1.ground_truth == stage2.ground_truth == "C"
    row = build_rl_row(stage2, config)
    assert row["data_source"] == "fixture_ds/stage2"
    assert row["extra_info"]["candidates"] == list(stage2.candidates)
    # per-source pairing: same ground truth, stage1 prompt uses full registry
    assert stage1.metadata == stage2.metadata
