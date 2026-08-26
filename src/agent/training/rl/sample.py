"""RL source-pair builder with a native-tool runtime episode.

The source export retains a Stage1/Stage2 pair for shared release validation,
but formal mixture materialization projects only the Stage1 row. That row is
a complete Qwen3.5 native-tool episode prompt; runtime never splices the
fixture Stage2 prompt into the trajectory.

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
import json
from pathlib import Path
from typing import Any, Mapping

from agent.task.contracts import CorpusCategory, GradingConfig, LeafRegistry, TaskConfig
from agent.task.prompt_choices import PromptChoiceRegistry
from agent.task.prompts import Prompt, build_stage1_prompt, build_stage2_prompt


NATIVE_TOOL_TRAJECTORY_FORMAT = "qwen3.5-native-tools-v2"
PROMPT_METADATA_FIELDS = (
    "field_name",
    "table_name",
    "field_description",
    "table_description",
)
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
    # Optional joint grading target.  Kept in task/sample data (not algorithm
    # state) so Stage 1 and Stage 2 rows share one immutable level label.
    ground_truth_level: str | None = None
    trajectory_format: str | None = None

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
        if self.ground_truth_level is not None and not self.ground_truth_level.strip():
            raise ValueError("ground_truth_level must be non-empty when provided")
        if self.trajectory_format is not None:
            if self.stage != "stage1":
                raise ValueError("trajectory_format is valid only for the runtime stage1 row")
            if self.trajectory_format != NATIVE_TOOL_TRAJECTORY_FORMAT:
                raise ValueError("unsupported RL trajectory_format")


def render_catalog_index(registry: LeafRegistry) -> str:
    """Render the compact "choice_id|display_name" catalog for in-context recall.

    The index is the primary recall surface: opaque choice ids keep canonical
    ids out of the prompt while letting the model match semantics directly,
    independent of any deterministic retriever's lexical ceiling.
    """

    choices = PromptChoiceRegistry.from_registry(registry).choices
    return "\n".join(f"{choice.choice_id}|{choice.display_name}" for choice in choices)


def build_native_tool_prompt(
    metadata: Mapping[str, str], grading: GradingConfig, registry: LeafRegistry
) -> Prompt:
    """Build the catalog-in-context prompt consumed by the native ToolAgentLoop.

    Metadata exposes both identity keys (field_name/table_name) and their
    human descriptions when present — the measured oracle-recall driver
    (3.2% -> 61.8% top-5 on real shougang records). Empty descriptions are
    normalized to empty strings by :func:`visible_metadata`.
    """

    if set(metadata) != set(PROMPT_METADATA_FIELDS):
        raise ValueError(
            "native tool prompt metadata must be " + "+".join(PROMPT_METADATA_FIELDS)
        )
    # Canonical key order: parquet round-trips reorder mapping keys, and the
    # prompt bytes must stay identical between export and validation.
    ordered = {field: str(metadata.get(field, "")) for field in PROMPT_METADATA_FIELDS}
    rubric = [[code, description] for code, description in grading.rubric()]
    system = (
        "You classify one database field into one category and assign a sensitivity level.\n"
        "The full category catalog is listed below as \"choice_id|name\" lines; recall "
        "candidates from this catalog directly using field and table semantics. When "
        "candidate names alone are ambiguous, call get_category_details or "
        "get_category_examples on the candidate ids to inspect definitions and samples. "
        "If the catalog seems insufficient you may call search_categories(field_name, "
        "table_name) once as a fallback hint. Make at most three tool calls total.\n"
        "Your terminal response must be exactly one JSON object with keys answer and "
        "level. answer must be an opaque choice_id from the catalog, never a category "
        "name or canonical id. level must be one approved sensitivity code. Do not "
        "output reasoning, Markdown, or extra keys.\n"
        "Approved sensitivity levels:\n"
        + json.dumps(rubric, ensure_ascii=False, separators=(",", ":"))
        + "\nCatalog (choice_id|name):\n"
        + render_catalog_index(registry)
    )
    user = (
        "Classify this field metadata:\n"
        + json.dumps(ordered, ensure_ascii=False, separators=(",", ":"))
    )
    return Prompt(system=system, user=user)


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
    grading: GradingConfig | None = None,
) -> tuple[RlSample, RlSample]:
    """Build the stage1 + stage2 RL samples for one canonical record.

    Only resolved records may be passed (the caller filters on
    resolution_status); the ground truth is validated as target.category_id
    with the registry as the final constraint. Stage 2 candidates reuse the
    deterministic fixture policy (GT + four non-GT registry ids, permuted
    deterministically by the stable source_id).
    """
    ground_truth = canonical_target(item, index, source, registry)
    if ground_truth is None:
        raise ValueError(f"item {index} in {source} is not resolved; cannot build RL samples")
    source_id = str(item.get("id", "") or "").strip()
    if not source_id:
        raise ValueError(f"item {index} in {source} has no stable id")
    metadata = visible_metadata(item.get("metadata", {}), config)
    candidates = tuple(build_candidates(ground_truth, registry, source_id=source_id))
    ground_truth_level: str | None = None
    if grading is not None:
        raw_level = item.get(grading.gt_field, "")
        ground_truth_level = "" if raw_level is None else str(raw_level).strip()
        if not ground_truth_level:
            raise ValueError(
                f"item {index} in {source} has no grading label under "
                f"{grading.gt_field!r}"
            )
        if ground_truth_level not in grading.levels:
            raise ValueError(
                f"item {index} in {source} has grading label {ground_truth_level!r} "
                f"outside configured levels {list(grading.levels)}"
            )
    stage1_prompt = (
        build_native_tool_prompt(metadata, grading, registry)
        if grading is not None
        else build_stage1_prompt(metadata, registry, config)
    )
    stage2_prompt = build_stage2_prompt(
        metadata,
        candidates,
        registry,
        config,
        corpus=corpus or None,
        grading=grading,
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
        ground_truth_level=ground_truth_level,
        trajectory_format=(
            NATIVE_TOOL_TRAJECTORY_FORMAT if grading is not None else None
        ),
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
        ground_truth_level=ground_truth_level,
    )
    return stage1, stage2


def build_rl_row(sample: RlSample, task_config: TaskConfig) -> dict[str, Any]:
    """Convert an RL sample to one VeRL v0.9.0 RL parquet row.

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
        "ground_truth_level": sample.ground_truth_level,
    }
    if sample.trajectory_format is not None:
        extra_info["trajectory_format"] = sample.trajectory_format
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
    "NATIVE_TOOL_TRAJECTORY_FORMAT",
    "PROMPT_METADATA_FIELDS",
    "build_native_tool_prompt",
    "render_catalog_index",
    "visible_metadata",
    "build_rl_samples",
    "build_rl_row",
]
