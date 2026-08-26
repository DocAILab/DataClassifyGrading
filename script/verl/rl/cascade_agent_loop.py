"""VeRL 0.8 AgentLoop adapter for the formal two-stage cascade.

The module remains importable in the CPU-only test environment.  When VeRL
is installed, :class:`DataClassifyCascadeAgentLoop` is registered as
``dataclassify_cascade`` and produces one causal trajectory:

    Stage 1 assistant tokens (loss mask 1)
    -> dynamic Stage 2 user bridge (loss mask 0)
    -> Stage 2 assistant tokens (loss mask 1)

Stage 2 candidates are decoded solely from that trajectory's Stage 1 output;
no oracle candidate bundle is read from the parquet row.
"""

from __future__ import annotations

from math import isfinite
from pathlib import Path
import time
from typing import Any, Mapping
from uuid import uuid4

from agent.task import LeafRegistry
from agent.task.assets import load_corpus_categories
from agent.task.grading_manifest import DatasetGradingManifest
from agent.task.prompts import Prompt
from agent.training.rl.cascade import (
    CascadeSource,
    build_cascade_stage2_prompt,
    cascade_reward,
    decode_stage1_candidates,
    decode_stage2_output,
)

try:
    from verl.experimental.agent_loop.agent_loop import (
        AgentLoopBase,
        AgentLoopMetrics,
        AgentLoopOutput,
        register,
    )
except ModuleNotFoundError as exc:  # normal CPU/unit environment
    if exc.name != "verl":
        raise
    VERL_AGENT_LOOP_AVAILABLE = False
    AgentLoopBase = object  # type: ignore[assignment,misc]
    AgentLoopMetrics = None  # type: ignore[assignment]
    AgentLoopOutput = None  # type: ignore[assignment]
    register = None  # type: ignore[assignment]
else:
    VERL_AGENT_LOOP_AVAILABLE = True


def build_stage2_bridge_text(prompt: Prompt) -> str:
    """Render the shared Stage 2 system+user contract as a user turn.

    A second system role in the middle of a chat is not portable across model
    templates.  The original shared system instruction is therefore preserved
    verbatim inside the dynamic user bridge.
    """

    return "Stage 2 instructions:\n" + prompt.system + "\n\n" + prompt.user


def _plain(value: Any) -> Any:
    """Normalize numpy scalar/object wrappers supplied by RLHFDataset."""

    if hasattr(value, "item") and not isinstance(value, (str, bytes, Mapping)):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass
    return value


def _mapping(value: Any, description: str) -> Mapping[str, Any]:
    value = _plain(value)
    if not isinstance(value, Mapping):
        raise ValueError(f"cascade AgentLoop requires {description} mapping")
    return value


def _source_from_kwargs(kwargs: Mapping[str, Any]) -> CascadeSource:
    extra = _mapping(kwargs.get("extra_info"), "extra_info")
    if extra.get("stage") != "stage1":
        raise ValueError("formal cascade parquet must contain Stage 1 rows only")
    unknown_extra = set(extra) - {
        "dataset", "stage", "source_id", "metadata", "ground_truth_level"
    }
    if unknown_extra:
        raise ValueError(
            "formal Stage 1 extra_info contains unexpected keys: "
            + ", ".join(sorted(str(key) for key in unknown_extra))
        )
    metadata = _mapping(extra.get("metadata"), "extra_info.metadata")
    if set(metadata) != {"field_name", "table_name"}:
        raise ValueError(
            "formal cascade prompt metadata must be field_name+table_name "
            "(contract change 2025-08-25)"
        )
    reward_model = _mapping(kwargs.get("reward_model"), "reward_model")
    if set(reward_model) != {"style", "ground_truth"} or reward_model.get("style") != "rule":
        raise ValueError("cascade reward_model must be the exact rule/ground_truth shape")
    ground_truth = reward_model.get("ground_truth")
    level = extra.get("ground_truth_level")
    source_id = extra.get("source_id")
    dataset = extra.get("dataset")
    if not all(isinstance(item, str) and item.strip() for item in (
        source_id, dataset, ground_truth, level, metadata.get("field_name")
    )):
        raise ValueError(
            "cascade source_id/dataset/field_name/ground_truth/ground_truth_level "
            "must be non-empty strings"
        )
    return CascadeSource(
        source_id=source_id.strip(),
        source_group=source_id.strip(),
        field_name=metadata["field_name"].strip(),
        ground_truth=ground_truth.strip(),
        ground_truth_level=level.strip(),
        dataset=dataset.strip(),
        metadata=dict(metadata),
    )


if not VERL_AGENT_LOOP_AVAILABLE:

    class DataClassifyCascadeAgentLoop:  # pragma: no cover - trivial local guard
        def __init__(self, *_: Any, **__: Any) -> None:
            raise RuntimeError(
                "DataClassifyCascadeAgentLoop requires the pinned VeRL 0.8 runtime"
            )

else:

    @register("dataclassify_cascade")
    class DataClassifyCascadeAgentLoop(AgentLoopBase):
        """Generate and score one direct Stage1→Stage2 trajectory."""

        def __init__(
            self,
            *args: Any,
            registry_path: str,
            corpus_path: str,
            grading_manifest_path: str,
            stage1_max_tokens: int = 96,
            stage2_max_tokens: int = 64,
            **kwargs: Any,
        ) -> None:
            super().__init__(*args, **kwargs)
            self.registry = LeafRegistry.from_path(Path(registry_path))
            missing_descriptions = sorted(
                category.category_id
                for category in self.registry.categories
                if not isinstance(category.description, str)
                or not category.description.strip()
            )
            if missing_descriptions:
                raise ValueError(
                    "formal cascade registry entries require non-empty descriptions: "
                    + ", ".join(missing_descriptions)
                )
            self.corpus = {
                item.category_id: item
                for item in load_corpus_categories(Path(corpus_path))
            }
            self.grading_manifest = DatasetGradingManifest.from_path(
                Path(grading_manifest_path)
            )
            invalid_rubrics = sorted(
                dataset
                for dataset in ("finance", "shougang")
                if self.grading_manifest.config_for(dataset).gt_field != "data_level"
            )
            if invalid_rubrics:
                raise ValueError(
                    "formal cascade grading gt_field must be data_level for: "
                    + ", ".join(invalid_rubrics)
                )
            missing = sorted(set(self.registry.ids) - set(self.corpus))
            if missing:
                raise ValueError(
                    f"cascade corpus is missing registry categories: {missing}"
                )
            if stage1_max_tokens < 1 or stage2_max_tokens < 1:
                raise ValueError("cascade per-stage token limits must be positive")
            self.stage1_max_tokens = int(stage1_max_tokens)
            self.stage2_max_tokens = int(stage2_max_tokens)
            self.response_length = int(self.rollout_config.response_length)
            if self.response_length < 1:
                raise ValueError("cascade rollout response_length must be positive")
            # The bridge is dynamic, so its exact cost is checked after Stage 1.
            # Still reject an impossible static budget up front rather than
            # relying on server-side truncation (which would silently alter the
            # causal trajectory and its loss mask).
            if self.stage1_max_tokens + self.stage2_max_tokens > self.response_length:
                raise ValueError(
                    "cascade per-stage token limits exceed rollout response_length"
                )

        async def _generate(
            self,
            request_id: str,
            prompt_ids: list[int],
            sampling_params: Mapping[str, Any],
            max_tokens: int,
        ) -> Any:
            params = dict(sampling_params)
            params["max_tokens"] = max_tokens
            # VeRL's LLMServerClient uses request_id for sticky routing.  A
            # direct cascade is one trajectory, therefore both generations
            # must use the same id; creating one UUID per turn can route Stage
            # 2 to a different replica and defeats the multi-turn contract.
            return await self.server_manager.generate(
                request_id=request_id,
                prompt_ids=prompt_ids,
                sampling_params=params,
            )

        @staticmethod
        def _checked_tokens(output: Any, *, stage: str, budget: int) -> list[int]:
            raw = getattr(output, "token_ids", None)
            if raw is None:
                raise ValueError(f"{stage} generation returned no token_ids")
            try:
                values = [int(token) for token in raw]
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{stage} generation returned invalid token_ids") from exc
            if len(values) > budget:
                raise ValueError(
                    f"{stage} generation returned {len(values)} tokens; budget is {budget}"
                )
            return values

        @staticmethod
        def _checked_logprobs(
            output: Any, *, stage: str, token_count: int
        ) -> list[float] | None:
            raw = getattr(output, "log_probs", None)
            if raw is None:
                return None
            try:
                raw_values = list(raw)
                if not raw_values:
                    return None
                values = [float(prob) for prob in raw_values]
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{stage} generation returned invalid log_probs") from exc
            if len(values) != token_count:
                raise ValueError(
                    f"{stage} log_probs length {len(values)} does not match "
                    f"token_ids length {token_count}"
                )
            if any(not isfinite(value) for value in values):
                raise ValueError(f"{stage} generation returned non-finite log_probs")
            return values

        async def run(
            self, sampling_params: dict[str, Any], **kwargs: Any
        ) -> AgentLoopOutput:
            source = _source_from_kwargs(kwargs)
            if source.ground_truth not in self.registry.ids:
                raise ValueError("cascade reward ground_truth is absent from the leaf registry")
            raw_prompt = _plain(kwargs.get("raw_prompt"))
            if not isinstance(raw_prompt, (list, tuple)) or not raw_prompt:
                raise ValueError("cascade AgentLoop requires non-empty raw_prompt messages")
            messages = [dict(_mapping(message, "raw_prompt message")) for message in raw_prompt]
            if [message.get("role") for message in messages] != ["system", "user"]:
                raise ValueError(
                    "formal cascade raw_prompt must contain exactly system and user messages"
                )
            prompt_ids = list(await self.apply_chat_template(messages))

            started = time.perf_counter()
            request_id = uuid4().hex
            stage1_output = await self._generate(
                request_id, prompt_ids, sampling_params, self.stage1_max_tokens
            )
            stage1_ids = self._checked_tokens(
                stage1_output, stage="Stage 1", budget=self.stage1_max_tokens
            )
            stage1_text = self.tokenizer.decode(stage1_ids, skip_special_tokens=True)
            candidates, stage1_errors = decode_stage1_candidates(
                stage1_text, registry=self.registry
            )
            stage1_logprobs = self._checked_logprobs(
                stage1_output, stage="Stage 1", token_count=len(stage1_ids)
            )
            raw_preempted = getattr(stage1_output, "num_preempted", None)
            stage1_preempted = (
                int(raw_preempted)
                if isinstance(raw_preempted, int) and raw_preempted >= 0
                else -1
            )

            if stage1_errors:
                metrics = AgentLoopMetrics(
                    generate_sequences=time.perf_counter() - started,
                    num_preempted=stage1_preempted,
                )
                return AgentLoopOutput(
                    prompt_ids=prompt_ids,
                    response_ids=stage1_ids,
                    response_mask=[1] * len(stage1_ids),
                    response_logprobs=stage1_logprobs,
                    reward_score=0.0,
                    num_turns=2,
                    metrics=metrics,
                    extra_fields={
                        "cascade_stage1_valid": False,
                        "cascade_termination": "invalid_stage1",
                        "cascade_stage1_hit": None,
                        "cascade_stage2_valid": None,
                        "cascade_leaf_correct": None,
                        "cascade_level_correct": None,
                        "cascade_candidates": [],
                        "turn_scores": [],
                        "tool_rewards": [],
                    },
                )

            grading = self.grading_manifest.config_for(source.dataset)
            if source.ground_truth_level not in grading.levels:
                raise ValueError(
                    f"ground_truth_level is outside the approved {source.dataset} rubric"
                )
            stage2_prompt = build_cascade_stage2_prompt(
                source,
                candidates,
                self.registry,
                self.corpus,
                grading=grading,
            )
            bridge_ids = list(
                await self.apply_chat_template(
                    [{"role": "user", "content": build_stage2_bridge_text(stage2_prompt)}],
                    remove_system_prompt=True,
                )
            )
            if not bridge_ids:
                raise ValueError("cascade Stage 2 bridge produced no token_ids")
            combined_prompt = prompt_ids + stage1_ids + bridge_ids
            if len(stage1_ids) + len(bridge_ids) + self.stage2_max_tokens > self.response_length:
                raise ValueError(
                    "cascade response_length cannot hold Stage 1, Stage 2 bridge, "
                    "and Stage 2 response"
                )
            stage2_output = await self._generate(
                request_id, combined_prompt, sampling_params, self.stage2_max_tokens
            )
            stage2_ids = self._checked_tokens(
                stage2_output, stage="Stage 2", budget=self.stage2_max_tokens
            )
            stage2_text = self.tokenizer.decode(stage2_ids, skip_special_tokens=True)
            stage2_logprobs = self._checked_logprobs(
                stage2_output, stage="Stage 2", token_count=len(stage2_ids)
            )
            leaf, level, stage2_errors = decode_stage2_output(
                stage2_text,
                candidates=candidates,
                grading=grading,
            )
            stage2_valid = not stage2_errors
            stage1_hit = source.ground_truth in candidates
            leaf_correct = stage2_valid and leaf == source.ground_truth
            level_correct = stage2_valid and level == source.ground_truth_level
            reward = cascade_reward(
                stage1_hit=stage1_hit,
                leaf_correct=leaf_correct,
                level_correct=level_correct,
                valid=stage2_valid,
            )

            response_ids = stage1_ids + bridge_ids + stage2_ids
            response_mask = (
                [1] * len(stage1_ids)
                + [0] * len(bridge_ids)
                + [1] * len(stage2_ids)
            )
            if stage1_logprobs is not None and stage2_logprobs is not None:
                response_logprobs = (
                    stage1_logprobs
                    + [0.0] * len(bridge_ids)
                    + stage2_logprobs
                )
            else:
                response_logprobs = None
            raw_stage2_preempted = getattr(stage2_output, "num_preempted", None)
            stage2_preempted = (
                int(raw_stage2_preempted)
                if isinstance(raw_stage2_preempted, int) and raw_stage2_preempted >= 0
                else -1
            )
            preempted = (
                stage1_preempted + stage2_preempted
                if stage1_preempted >= 0 and stage2_preempted >= 0
                else -1
            )
            metrics = AgentLoopMetrics(
                generate_sequences=time.perf_counter() - started,
                num_preempted=preempted,
            )
            return AgentLoopOutput(
                prompt_ids=prompt_ids,
                response_ids=response_ids,
                response_mask=response_mask,
                response_logprobs=response_logprobs,
                reward_score=reward,
                num_turns=4,
                metrics=metrics,
                extra_fields={
                    "cascade_stage1_valid": True,
                    "cascade_termination": "completed",
                    "cascade_stage1_hit": stage1_hit,
                    "cascade_stage2_valid": stage2_valid,
                    "cascade_leaf_correct": leaf_correct,
                    "cascade_level_correct": level_correct,
                    "cascade_candidates": list(candidates),
                    "turn_scores": [],
                    "tool_rewards": [],
                },
            )


__all__ = [
    "VERL_AGENT_LOOP_AVAILABLE",
    "DataClassifyCascadeAgentLoop",
    "build_stage2_bridge_text",
]
