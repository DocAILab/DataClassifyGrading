"""VeRL 0.8 compatibility seam for the RL parquet contract.

The compatibility job installs the pinned VeRL wheel and runs this module;
ordinary CPU CI skips only the backend-specific test while retaining the
contract/reward tests below.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import inspect
import json
from pathlib import Path

import pytest

from agent.task import GradingConfig, LeafRegistry, TaskConfig
from agent.task.assets import load_corpus_categories
from agent.training.rl import export_rl_dataset, validate_rl_dataset
from agent.training.rl.reward import reward_stage2_choices

pytestmark = pytest.mark.verl
ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "sft" / "fixtures"


def _assets():
    registry = LeafRegistry.from_path(FIXTURES / "registry.json")
    task = TaskConfig.from_path(FIXTURES / "task.json")
    corpus = {
        category.category_id: category
        for category in load_corpus_categories(FIXTURES / "corpus.json")
    }
    grading = GradingConfig.from_path(FIXTURES / "grading.json")
    return registry, task, corpus, grading


def _export(tmp_path: Path) -> Path:
    registry, task, corpus, grading = _assets()
    output = tmp_path / "rl"
    report = export_rl_dataset(
        FIXTURES / "canonical" / "all.json",
        None,
        output,
        "demo",
        registry,
        task,
        corpus=corpus,
        grading=grading,
    )
    assert report["label_gap_gate"]["status"] == "passed"
    validation = validate_rl_dataset(
        output,
        "demo",
        registry,
        task,
        corpus=corpus,
        grading=grading,
    )
    assert validation["valid"], validation
    return output


def test_rl_reward_rejects_half_configured_joint_context() -> None:
    registry, _, _, grading = _assets()
    candidates = tuple(registry.ids[:5])
    with pytest.raises(ValueError, match="together"):
        reward_stage2_choices(
            '{"answer":"1","level":"L1"}',
            ground_truth=candidates[0],
            candidates=candidates,
            registry=registry,
            grading=grading,
        )
    with pytest.raises(ValueError, match="together"):
        reward_stage2_choices(
            '{"answer":"1"}',
            ground_truth=candidates[0],
            candidates=candidates,
            registry=registry,
            expected_level="L1",
        )


def test_rl_reward_joint_context_requires_approved_level() -> None:
    registry, _, _, grading = _assets()
    candidates = tuple(registry.ids[:5])
    result = reward_stage2_choices(
        '{"answer":"1","level":"L1"}',
        ground_truth=candidates[0],
        candidates=candidates,
        registry=registry,
        grading=grading,
        expected_level="L1",
    )
    assert result.reward == 1.0


def test_exported_rl_parquet_has_verl_five_field_contract(tmp_path: Path) -> None:
    # This test is backend-independent but exercises the exact artifact that
    # the VeRL compatibility job consumes.
    import pyarrow.parquet as pq

    output = _export(tmp_path)
    rows = pq.read_table(output / "train.parquet").to_pylist()
    assert rows
    assert set(rows[0]) == {
        "data_source",
        "prompt",
        "ability",
        "reward_model",
        "extra_info",
    }
    stage2 = next(row for row in rows if row["extra_info"]["stage"] == "stage2")
    assert stage2["extra_info"]["ground_truth_level"] in {"L1", "L2", "L3", "L4"}
    assert len(stage2["extra_info"]["candidates"]) == 5


@pytest.mark.verl
def test_official_agent_loop_api_and_reward_tensor_probe() -> None:
    """Probe VeRL 0.8's CPU-visible dataclasses; never starts Ray or CUDA."""

    pytest.importorskip("verl")
    assert importlib.metadata.version("verl") == "0.8.0"
    from verl.experimental.agent_loop.agent_loop import (
        AgentLoopBase,
        AgentLoopMetrics,
        AgentLoopOutput,
    )
    from verl.workers.rollout.llm_server import LLMServerClient
    from verl.workers.rollout.replica import TokenOutput

    init_parameters = inspect.signature(AgentLoopBase.__init__).parameters
    assert {
        "trainer_config",
        "server_manager",
        "tokenizer",
        "processor",
        "dataset_cls",
        "data_config",
    }.issubset(init_parameters)
    generate_parameters = inspect.signature(LLMServerClient.generate).parameters
    assert {"request_id", "prompt_ids", "sampling_params"}.issubset(generate_parameters)
    assert {"token_ids", "log_probs", "num_preempted"}.issubset(
        inspect.signature(TokenOutput).parameters
    )

    output = AgentLoopOutput(
        prompt_ids=[10],
        response_ids=[20, 21],
        response_mask=[1, 0],
        response_logprobs=[-0.1, 0.0],
        reward_score=0.75,
        num_turns=4,
        metrics=AgentLoopMetrics(generate_sequences=0.0, num_preempted=0),
    )
    payload = output.as_dict()
    assert payload["responses"].tolist() == [20, 21]
    assert payload["response_mask"].tolist() == [1, 0]
    assert payload["rollout_log_probs"].tolist() == [-0.1, 0.0]
    assert payload["rm_scores"].tolist() == [0.0, 0.75]


def test_agent_loop_config_loads_as_named_hydra_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("verl")
    from omegaconf import OmegaConf

    for name, value in {
        "DATACLASSIFY_RLOO_REGISTRY": str(FIXTURES / "registry.json"),
        "DATACLASSIFY_RLOO_CORPUS": str(FIXTURES / "corpus.json"),
        "DATACLASSIFY_RLOO_GRADING_MANIFEST": str(FIXTURES / "grading_manifest.json"),
    }.items():
        monkeypatch.setenv(name, value)
    configs = OmegaConf.load(ROOT / "cfg" / "verl" / "rl" / "cascade_agent_loop.yaml")
    assert OmegaConf.is_list(configs)
    assert configs[0].name == "dataclassify_cascade"
    assert configs[0]._target_ == (
        "script.verl.rl.cascade_agent_loop.DataClassifyCascadeAgentLoop"
    )


def test_cascade_agent_loop_uses_official_verl_08_interface() -> None:
    pytest.importorskip("verl")
    assert importlib.metadata.version("verl") == "0.8.0"
    from verl.experimental.agent_loop.agent_loop import AgentLoopBase
    from script.verl.rl.cascade_agent_loop import (
        DataClassifyCascadeAgentLoop,
        VERL_AGENT_LOOP_AVAILABLE,
    )

    assert VERL_AGENT_LOOP_AVAILABLE
    assert issubclass(DataClassifyCascadeAgentLoop, AgentLoopBase)
    config_text = (
        ROOT / "cfg" / "verl" / "rl" / "cascade_agent_loop.yaml"
    ).read_text(encoding="utf-8")
    assert "script.verl.rl.cascade_agent_loop.DataClassifyCascadeAgentLoop" in config_text
    assert "DATACLASSIFY_RLOO_GRADING_MANIFEST" in config_text


@pytest.mark.verl
def test_rl_parquet_loads_through_verl_08_rlhfdataset(tmp_path: Path) -> None:
    # The regular CPU suite has no VeRL; the dedicated compat job installs
    # requirements/verl.txt and turns this into a real backend load test.
    pytest.importorskip("verl")
    assert importlib.metadata.version("verl") == "0.8.0"
    module = importlib.import_module("verl.utils.dataset.rl_dataset")
    dataset_cls = getattr(module, "RLHFDataset")
    output = _export(tmp_path)

    # VeRL 0.8 has used both ``data_files`` and ``parquet_files`` names in
    # nearby patch releases.  Inspect the installed class rather than hiding
    # a version-specific constructor behind a fake import.
    params = inspect.signature(dataset_cls).parameters
    kwargs: dict[str, object] = {}
    if "data_files" in params:
        kwargs["data_files"] = [str(output / "train.parquet")]
    elif "parquet_files" in params:
        kwargs["parquet_files"] = [str(output / "train.parquet")]
    else:
        raise AssertionError("VeRL RLHFDataset has no parquet file argument")
    if "tokenizer" in params:
        kwargs["tokenizer"] = None
    if "config" in params:
        kwargs["config"] = {
            "prompt_key": "prompt",
            "reward_fn_key": "data_source",
            "max_prompt_length": 4096,
            "truncation": "error",
            "filter_prompts": False,
            "filter_overlong_prompts": False,
            "return_raw_chat": True,
        }
    dataset = dataset_cls(**kwargs)
    assert len(dataset) == 6
