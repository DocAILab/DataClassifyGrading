"""RL sample contract and builder for the two-stage classification task.

A sample is the unit consumed by future RL training: prompt messages,
ground truth, task metadata and reward metadata — never training-algorithm
internal state (no advantage estimates, no rollout fields, no policy
state).

Ground truth is EXCLUSIVELY ``target.category_id`` from the canonical
dataset contract (resolution_status == "resolved"); classification
level_1..level_4 stay provenance and are deliberately not consulted.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from agent.task.contracts import CorpusCategory, LeafRegistry, TaskConfig
from agent.task.prompts import build_stage1_prompt, build_stage2_prompt
from agent.training.common import build_candidates, canonical_target


@dataclass(frozen=True)
class RlMessage:
    """One chat message of an RL prompt (system/user only, never assistant)."""

    role: str
    content: str

    def __post_init__(self) -> None:
        if self.role not in {"system", "user"}:
            raise ValueError(f"RL prompt messages must be system or user, got {self.role!r}")
        if not isinstance(self.content, str):
            raise ValueError("RL prompt message content must be a string")

    def to_mapping(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True)
class RewardMeta:
    """Reward routing metadata: which task stage a sample belongs to.

    Contains task facts only; reward weights and algorithm logic stay out.
    """

    dataset: str
    stage: str

    def __post_init__(self) -> None:
        if not self.dataset.strip():
            raise ValueError("reward metadata dataset must be non-empty")
        if self.stage not in {"stage1", "stage2"}:
            raise ValueError(f"reward metadata stage must be stage1 or stage2, got {self.stage!r}")


@dataclass(frozen=True)
class RlSample:
    """One RL sample of the two-stage classification task.

    - stage: "stage1" (retrieve 5 candidates from the full registry) or
      "stage2" (pick one answer from the candidate bundle).
    - source_id: stable record id from the canonical dataset.
    - messages: system + user prompt; there is deliberately NO assistant
      gold response (rollout generates it).
    - ground_truth: target.category_id (the only training label).
    - candidates: the deterministic Stage 2 candidate bundle
      (Stage 2 only); None for Stage 1. This is a fixture policy, NOT the
      production retrieval policy.
    - metadata: prompt-visible task metadata fields (task config).
    - reward: reward routing metadata (dataset + stage).
    """

    stage: str
    dataset: str
    source_id: str
    messages: tuple[RlMessage, ...]
    ground_truth: str
    candidates: tuple[str, ...] | None
    metadata: Mapping[str, str]
    reward: RewardMeta

    def __post_init__(self) -> None:
        if self.stage not in {"stage1", "stage2"}:
            raise ValueError(f"sample stage must be stage1 or stage2, got {self.stage!r}")
        if not isinstance(self.source_id, str) or not self.source_id.strip():
            raise ValueError("sample source_id must be a non-empty string")
        if len(self.messages) != 2 or [message.role for message in self.messages] != ["system", "user"]:
            raise ValueError("RL sample messages must be exactly [system, user]")
        if not isinstance(self.ground_truth, str) or not self.ground_truth.strip():
            raise ValueError("sample ground_truth must be a non-empty category_id")
        if self.stage == "stage2":
            if self.candidates is None:
                raise ValueError("stage2 sample requires a candidate bundle")
            if len(self.candidates) != 5 or len(set(self.candidates)) != 5:
                raise ValueError("stage2 sample candidates must be exactly 5 unique category_ids")
        elif self.candidates is not None:
            raise ValueError("stage1 sample must not carry a candidate bundle")


def visible_metadata(metadata: Mapping[str, Any], config: TaskConfig) -> dict[str, str]:
    """Select the task-config metadata fields with None normalized to empty."""
    if not isinstance(metadata, Mapping):
        raise ValueError("record metadata must be an object")
    return {
        field: ("" if metadata.get(field) is None else str(metadata.get(field, "")))
        for field in config.metadata_fields
    }


def build_rl_samples(
    item: Mapping[str, Any],
    index: int,
    source: Path,
    *,
    dataset: str,
    registry: LeafRegistry,
    config: TaskConfig,
    corpus: Mapping[str, CorpusCategory],
) -> tuple[RlSample, RlSample]:
    """Build the stage1 + stage2 RL samples for one canonical record.

    Only resolved records may be passed (the caller filters on
    resolution_status); the ground truth is validated as target.category_id
    with the registry as the final constraint. Stage 2 candidates reuse the
    deterministic fixture policy (GT + first four non-GT registry IDs).
    """
    ground_truth = canonical_target(item, index, source, registry)
    if ground_truth is None:
        raise ValueError(f"item {index} in {source} is not resolved; cannot build RL samples")
    source_id = str(item.get("id", "") or "").strip()
    if not source_id:
        raise ValueError(f"item {index} in {source} has no stable id")
    metadata = visible_metadata(item.get("metadata", {}), config)
    candidates = tuple(build_candidates(ground_truth, registry))
    stage1_prompt = build_stage1_prompt(metadata, registry, config)
    stage2_prompt = build_stage2_prompt(
        metadata, candidates, registry, config, corpus=corpus or None
    )
    stage1 = RlSample(
        stage="stage1",
        dataset=dataset,
        source_id=source_id,
        messages=(
            RlMessage("system", stage1_prompt.system),
            RlMessage("user", stage1_prompt.user),
        ),
        ground_truth=ground_truth,
        candidates=None,
        metadata=metadata,
        reward=RewardMeta(dataset=dataset, stage="stage1"),
    )
    stage2 = RlSample(
        stage="stage2",
        dataset=dataset,
        source_id=source_id,
        messages=(
            RlMessage("system", stage2_prompt.system),
            RlMessage("user", stage2_prompt.user),
        ),
        ground_truth=ground_truth,
        candidates=candidates,
        metadata=metadata,
        reward=RewardMeta(dataset=dataset, stage="stage2"),
    )
    return stage1, stage2


def build_rl_row(sample: RlSample, task_config: TaskConfig) -> dict[str, Any]:
    """Convert an RL sample to one VeRL v0.8.0 RL parquet row.

    Five fields exactly (data_source / prompt / ability / reward_model /
    extra_info); ``prompt`` carries only system+user (rollout appends the
    generation prompt). ``reward_model`` follows the VeRL rule-reward
    convention (only ground_truth is consumed by VeRL builtin managers).
    """
    data_source = f"{sample.dataset}/{sample.stage}"
    extra_info: dict[str, Any] = {
        "dataset": sample.dataset,
        "stage": sample.stage,
        "source_id": sample.source_id,
        "metadata": dict(sample.metadata),
    }
    if sample.stage == "stage2":
        extra_info["candidates"] = list(sample.candidates)
    return {
        "data_source": data_source,
        "prompt": [message.to_mapping() for message in sample.messages],
        "ability": task_config.task_name,
        "reward_model": {
            "style": "rule",
            "ground_truth": sample.ground_truth,
        },
        "extra_info": extra_info,
    }


__all__ = [
    "RlMessage",
    "RewardMeta",
    "RlSample",
    "visible_metadata",
    "build_rl_samples",
    "build_rl_row",
]
