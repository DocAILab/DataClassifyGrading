"""Capture actual F66 executor messages, not a substitute hand-written prompt."""
import json
import pytest
from program_loop import _vendor
from agent.orchestration.contracts import MetadataRecord, ToolCall
from agent.orchestration.category_catalog import CategoryCatalog
from agent.orchestration.skill_loader import SkillLoader
from pool_catalog import pool_catalog, SEARCH_NAMES
from pool_metadata import PoolMetadata
from pool_runtime import PoolExecutor
from agent.orchestration.model_skill_executor import SkillExecutionError

REGISTRY = {f"cat{i}": {"label": f"Synthetic {i}", "description": f"Supplied standard {i}", "grading_level": "L2"}
            for i in range(1, 7)}


@pytest.mark.parametrize("name", ["SELECT_CATEGORY_TOPK", *sorted(SEARCH_NAMES), "INFER_CATEGORY_FROM_TOPK", "INFER_CATEGORY"])
def test_actual_provider_request_has_one_consistent_shape(name):
    captured = []
    retrieval = name in SEARCH_NAMES or name == "SELECT_CATEGORY_TOPK"
    def provider(messages):
        captured.append(messages)
        return json.dumps({"option_ids": [f"C00{i}" for i in range(1, 6)], "summary": "synthetic"}
                          if retrieval else {"option_id": "C001", "summary": "synthetic"})
    catalog = CategoryCatalog(REGISTRY)
    loader = SkillLoader(_vendor.VENDOR / "runtime/skills/full_pool", pool_catalog("F66"), strict_catalog=False)
    executor = PoolExecutor(provider, loader, PoolMetadata([]), catalog)
    state = {"standard_candidates": catalog.labels()[:5]}
    call = ToolCall("SELECT_CATEGORY_TOPK" if retrieval else "INFER_CATEGORY_FROM_TOPK", {})
    resolved = {"semantic_query": "explicit synthetic pre-observed semantics"} if retrieval else {
        "candidate_standard_categories": state["standard_candidates"], "all_available_evidence": {"synthetic": True}}
    record = MetadataRecord.from_mapping({"domain": "finance", "field_name": "CPU_FIELD", "source_id": "CPU_ONLY"})
    output = executor._execute_two_stage(call, record, resolved, state, loader.load(name))
    assert output.status == "success" and len(captured) == 1
    system = captured[0][0]["content"]
    assert "Data keys must be exactly" not in system
    assert "data.category" not in system
    assert 'Return exactly {"summary"' not in system
    assert '"option_ids"' in system if retrieval else '"option_id"' in system


def test_formal_comparison_provider_contract_is_already_consistent():
    loader = SkillLoader(_vendor.VENDOR / "runtime/skills/full_pool", pool_catalog("F66"), strict_catalog=False)
    body = loader.load("COMPARE_STANDARD_CANDIDATES").body
    assert "`decision`, `option_id`, and `reason`" in body
    assert "Do not emit" in body and "summary/data wrapper" in body


def test_explicit_insufficiency_is_not_retried_as_a_json_error():
    requests = []
    def provider(messages):
        requests.append(messages)
        return json.dumps({"option_id": None, "summary": "subject ownership remains unknown"})
    catalog = CategoryCatalog(REGISTRY)
    loader = SkillLoader(_vendor.VENDOR / "runtime/skills/full_pool", pool_catalog("F66"), strict_catalog=False)
    executor = PoolExecutor(provider, loader, PoolMetadata([]), catalog)
    state = {"standard_candidates": catalog.labels()[:5]}
    resolved = {"candidate_standard_categories": state["standard_candidates"], "all_available_evidence": {"synthetic": True}}
    record = MetadataRecord.from_mapping({"domain": "finance", "field_name": "CPU_FIELD", "source_id": "CPU_ONLY"})
    with pytest.raises(SkillExecutionError) as error:
        executor._execute_two_stage(ToolCall("INFER_CATEGORY_FROM_TOPK", {}), record, resolved, state, loader.load("INFER_CATEGORY"))
    assert error.value.metadata.get("failure_kind") == "insufficient_evidence"
    assert len(requests) == 1
    assert "category" not in state and state["standard_candidates"] == catalog.labels()[:5]
