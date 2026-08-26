"""Formal shougang field-only prompt identifiability gate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.training.input_audit import (
    audit_prompt_target_bundle,
    audit_prompt_target_conflicts,
    require_identifiable_prompt_bundle,
    require_identifiable_prompts,
)
from script.analysis.audit_prompt_conflicts import main


CLASSIFICATION_SHA = "a" * 64
GRADING_SHA = "b" * 64


def _record(record_id: str, field: str, leaf: str, level: str) -> dict:
    return {
        "id": record_id,
        "resolution_status": "resolved",
        "split": "train",
        "metadata": {"field_name": field, "table_name": "must-not-appear"},
        "target": {"category_id": leaf},
        "data_level": level,
    }


def test_same_prompt_key_with_multiple_joint_targets_is_blocking_and_redacted() -> None:
    records = [
        _record("source-one", " User_ID ", "leaf:a", "L2"),
        _record("source-two", "user_id", "leaf:b", "L3"),
    ]
    report = audit_prompt_target_conflicts(
        records,
        classification_standard_sha256=CLASSIFICATION_SHA,
        grading_standard_sha256=GRADING_SHA,
    )
    assert report["dataset"] == "shougang"
    assert report["status"] == "failed"
    assert report["conflict_keys"] == 1
    assert report["conflicting_records"] == 2
    assert report["conflicts"][0]["target_count"] == 2
    rendered = json.dumps(report, ensure_ascii=False)
    for forbidden in ("User_ID", "user_id", "leaf:a", "leaf:b", "source-one"):
        assert forbidden not in rendered
    with pytest.raises(ValueError, match="prompt-identifiability.*not identifiable"):
        require_identifiable_prompts(report)


def test_duplicate_prompt_with_same_target_is_identifiable() -> None:
    records = [
        _record("one", "acct_no", "leaf:a", "L2"),
        _record("two", "ACCT_NO", "leaf:a", "L2"),
    ]
    report = audit_prompt_target_conflicts(
        records,
        classification_standard_sha256=CLASSIFICATION_SHA,
        grading_standard_sha256=GRADING_SHA,
    )
    assert report["status"] == "passed"
    assert report["conflict_keys"] == 0
    require_identifiable_prompts(report)


def test_nonformal_dataset_is_rejected() -> None:
    with pytest.raises(ValueError, match="formal dataset"):
        audit_prompt_target_conflicts(
            [_record("one", "id", "leaf:a", "L1")],
            classification_standard_sha256=CLASSIFICATION_SHA,
            grading_standard_sha256=GRADING_SHA,
            dataset="finance",
        )


def test_different_standard_hashes_are_different_prompt_contracts() -> None:
    first = audit_prompt_target_conflicts(
        [_record("one", "id", "leaf:a", "L1")],
        classification_standard_sha256=CLASSIFICATION_SHA,
        grading_standard_sha256=GRADING_SHA,
    )
    second = audit_prompt_target_conflicts(
        [_record("two", "id", "leaf:b", "L4")],
        classification_standard_sha256="c" * 64,
        grading_standard_sha256=GRADING_SHA,
    )
    assert first["prompt_key_sha256"] != second["prompt_key_sha256"]


def test_cli_writes_redacted_blocking_report(tmp_path: Path) -> None:
    canonical = tmp_path / "all.json"
    canonical.write_text(
        json.dumps(
            [
                _record("secret-one", "Secret_Field", "leaf:a", "L1"),
                _record("secret-two", "secret_field", "leaf:b", "L2"),
            ]
        ),
        encoding="utf-8",
    )
    classification = tmp_path / "registry.json"
    grading = tmp_path / "grading.json"
    classification.write_text('{"standard":"classification"}', encoding="utf-8")
    grading.write_text('{"standard":"grading"}', encoding="utf-8")
    report = tmp_path / "audit.json"
    assert main(
        [
            "--canonical",
            str(canonical),
            "--classification-standard",
            str(classification),
            "--grading-standard",
            str(grading),
            "--report",
            str(report),
        ]
    ) == 1
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["dataset"] == "shougang"
    assert payload["status"] == "failed"
    rendered = report.read_text(encoding="utf-8")
    assert "Secret_Field" not in rendered
    assert "secret-one" not in rendered


def test_missing_joint_contract_fields_fail_fast() -> None:
    broken = _record("one", "", "leaf:a", "L1")
    with pytest.raises(ValueError, match="field_name"):
        audit_prompt_target_conflicts(
            [broken],
            classification_standard_sha256=CLASSIFICATION_SHA,
            grading_standard_sha256=GRADING_SHA,
        )


def test_shougang_bundle_has_single_clean_redacted_audit() -> None:
    shougang = [_record("shougang-one", "shougang_field", "leaf:b", "L2")]
    bundle = audit_prompt_target_bundle(
        {"shougang": shougang},
        standards_by_dataset={
            "shougang": {
                "classification_standard_sha256": CLASSIFICATION_SHA,
                "grading_standard_sha256": GRADING_SHA,
            }
        },
    )
    require_identifiable_prompt_bundle(bundle)
    assert bundle["status"] == "passed"
    assert set(bundle["datasets"]) == {"shougang"}
    assert set(bundle["standard_hashes"]) == {"shougang"}
    rendered = json.dumps(bundle, ensure_ascii=False)
    for forbidden in ("shougang_field", "leaf:b", "shougang-one"):
        assert forbidden not in rendered

    broken = json.loads(json.dumps(bundle))
    broken["datasets"]["shougang"]["status"] = "failed"
    with pytest.raises(ValueError, match="not passed|not identifiable"):
        require_identifiable_prompt_bundle(broken)


def test_bundle_rejects_two_dataset_inputs() -> None:
    records = [_record("one", "id", "leaf:a", "L1")]
    with pytest.raises(ValueError, match="formal dataset set"):
        audit_prompt_target_bundle(
            {"finance": records, "shougang": records},
            standards_by_dataset={
                "finance": {
                    "classification_standard_sha256": CLASSIFICATION_SHA,
                    "grading_standard_sha256": GRADING_SHA,
                },
                "shougang": {
                    "classification_standard_sha256": CLASSIFICATION_SHA,
                    "grading_standard_sha256": GRADING_SHA,
                },
            },
        )


def test_bundle_cli_audits_single_dataset_without_raw_values(tmp_path: Path) -> None:
    shougang = tmp_path / "shougang.json"
    shougang.write_text(
        json.dumps([_record("s-secret", "S_secret", "leaf:b", "L2")]),
        encoding="utf-8",
    )
    classification = tmp_path / "classification.json"
    grading = tmp_path / "grading.json"
    classification.write_text('{"standard":"classification"}', encoding="utf-8")
    grading.write_text('{"standard":"grading"}', encoding="utf-8")
    output = tmp_path / "bundle.json"
    assert main(
        [
            "--bundle",
            "--canonical",
            f"shougang={shougang}",
            "--classification-standard",
            f"shougang={classification}",
            "--grading-standard",
            f"shougang={grading}",
            "--report",
            str(output),
        ]
    ) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    require_identifiable_prompt_bundle(payload)
    rendered = output.read_text(encoding="utf-8")
    assert "S_secret" not in rendered and "s-secret" not in rendered
