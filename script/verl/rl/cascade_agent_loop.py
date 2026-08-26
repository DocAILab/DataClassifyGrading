"""VeRL 0.9 native-tool AgentLoop for the formal shougang trajectory.

The implementation delegates message rendering, Qwen3.5 tool-call parsing,
tool execution, sticky request routing and policy/observation masks to VeRL's
official :class:`ToolAgentLoop`.  This task adapter adds only fail-closed
source validation and one strict terminal category+level exact-match reward.
There is no synthetic Stage-2 user bridge.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from agent.release_policy import FORMAL_RELEASE_NAME
from agent.task import LeafRegistry
from agent.task.assets import load_corpus_categories
from agent.task.grading_manifest import DatasetGradingManifest
from agent.training.rl.cascade import CascadeSource
from agent.training.rl.native_tools import (
    exact_tool_reward,
    parse_final_tool_answer,
)
from agent.training.rl.sample import NATIVE_TOOL_TRAJECTORY_FORMAT, PROMPT_METADATA_FIELDS

try:
    from verl.experimental.agent_loop.agent_loop import register
    from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop
except ModuleNotFoundError as exc:  # normal CPU/unit environment
    if not exc.name or not exc.name.startswith("verl"):
        raise
    VERL_AGENT_LOOP_AVAILABLE = False
    ToolAgentLoop = object  # type: ignore[assignment,misc]
    register = None  # type: ignore[assignment]
else:
    VERL_AGENT_LOOP_AVAILABLE = True

_EXPECTED_TOOLS = {
    "search_categories",
    "get_category_details",
    "get_category_examples",
}


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
        raise ValueError(f"native tool AgentLoop requires {description} mapping")
    return value


def _source_from_kwargs(kwargs: Mapping[str, Any]) -> CascadeSource:
    extra = _mapping(kwargs.get("extra_info"), "extra_info")
    allowed_extra = {
        "dataset",
        "stage",
        "source_id",
        "metadata",
        "ground_truth_level",
        "trajectory_format",
    }
    unknown_extra = set(extra) - allowed_extra
    if unknown_extra:
        raise ValueError(
            "formal native-tool extra_info contains unexpected keys: "
            + ", ".join(sorted(str(key) for key in unknown_extra))
        )
    if extra.get("stage") != "stage1":
        raise ValueError("formal native-tool parquet must contain one episode row per source")
    if extra.get("trajectory_format") != NATIVE_TOOL_TRAJECTORY_FORMAT:
        raise ValueError(
            f"formal trajectory_format must be {NATIVE_TOOL_TRAJECTORY_FORMAT}"
        )
    metadata = _mapping(extra.get("metadata"), "extra_info.metadata")
    if set(metadata) != set(PROMPT_METADATA_FIELDS):
        raise ValueError(
            "formal native-tool metadata must be exactly "
            + "+".join(PROMPT_METADATA_FIELDS)
        )
    reward_model = _mapping(kwargs.get("reward_model"), "reward_model")
    if set(reward_model) != {"style", "ground_truth"} or reward_model.get("style") != "rule":
        raise ValueError("native-tool reward_model must be the exact rule/ground_truth shape")
    source_id = extra.get("source_id")
    dataset = extra.get("dataset")
    ground_truth = reward_model.get("ground_truth")
    level = extra.get("ground_truth_level")
    field_name = metadata.get("field_name")
    table_name = metadata.get("table_name")
    if dataset != FORMAL_RELEASE_NAME:
        raise ValueError(f"formal native-tool AgentLoop accepts only {FORMAL_RELEASE_NAME}")
    required = (source_id, dataset, ground_truth, level, field_name)
    if not all(isinstance(item, str) and item.strip() for item in required):
        raise ValueError(
            "source_id/dataset/field_name/ground_truth/ground_truth_level "
            "must be non-empty strings"
        )
    if not isinstance(table_name, str):
        raise ValueError("metadata.table_name must be a string")
    return CascadeSource(
        source_id=source_id.strip(),
        source_group=source_id.strip(),
        field_name=field_name.strip(),
        ground_truth=ground_truth.strip(),
        ground_truth_level=level.strip(),
        dataset=dataset.strip(),
        metadata={field: metadata.get(field, "") for field in PROMPT_METADATA_FIELDS},
    )


def final_policy_token_ids(
    response_ids: Sequence[int], response_mask: Sequence[int]
) -> tuple[int, ...]:
    """Return the final contiguous assistant segment, or empty after a tool turn."""

    if len(response_ids) != len(response_mask):
        raise ValueError("response ids and mask must have equal length")
    if any(mask not in (0, 1, False, True) for mask in response_mask):
        raise ValueError("response mask entries must be 0/1")
    if not response_mask or not bool(response_mask[-1]):
        return ()
    start = len(response_mask) - 1
    while start > 0 and bool(response_mask[start - 1]):
        start -= 1
    return tuple(int(token) for token in response_ids[start:])


if not VERL_AGENT_LOOP_AVAILABLE:

    class DataClassifyCascadeAgentLoop:  # pragma: no cover - local guard
        def __init__(self, *_: Any, **__: Any) -> None:
            raise RuntimeError(
                "DataClassifyCascadeAgentLoop requires VeRL 0.9.0 native ToolAgentLoop"
            )

else:

    @register("dataclassify_cascade")
    class DataClassifyCascadeAgentLoop(ToolAgentLoop):
        """Official ToolAgentLoop plus formal task validation and exact reward."""

        def __init__(
            self,
            *args: Any,
            registry_path: str,
            corpus_path: str,
            grading_manifest_path: str,
            **kwargs: Any,
        ) -> None:
            super().__init__(*args, **kwargs)
            self.registry = LeafRegistry.from_path(Path(registry_path))
            self.corpus = {
                item.category_id: item
                for item in load_corpus_categories(Path(corpus_path))
            }
            missing = sorted(set(self.registry.ids) - set(self.corpus))
            extra = sorted(set(self.corpus) - set(self.registry.ids))
            if missing or extra:
                raise ValueError(
                    "native-tool corpus must exactly cover the registry; "
                    f"missing={missing}, extra={extra}"
                )
            self.grading_manifest = DatasetGradingManifest.from_path(
                Path(grading_manifest_path)
            )
            grading = self.grading_manifest.config_for(FORMAL_RELEASE_NAME)
            if grading.gt_field != "data_level":
                raise ValueError("formal shougang grading gt_field must be data_level")
            if set(self.tools) != _EXPECTED_TOOLS:
                raise ValueError(
                    "formal native-tool environment must contain exactly: "
                    + ", ".join(sorted(_EXPECTED_TOOLS))
                )
            if self.max_parallel_calls != 1:
                raise ValueError("formal native-tool calls must be sequential")
            if self.max_assistant_turns is None or self.max_assistant_turns != 4:
                raise ValueError("formal native-tool max_assistant_turns must be 4")
            if self.max_user_turns is None or self.max_user_turns != 3:
                raise ValueError("formal native-tool max_user_turns must be 3")

        async def run(self, sampling_params: dict[str, Any], **kwargs: Any) -> Any:
            source = _source_from_kwargs(kwargs)
            if source.ground_truth not in self.registry.ids:
                raise ValueError("native-tool ground_truth is absent from the registry")
            grading = self.grading_manifest.config_for(source.dataset)
            if source.ground_truth_level not in grading.levels:
                raise ValueError("ground_truth_level is outside the approved rubric")
            raw_prompt = _plain(kwargs.get("raw_prompt"))
            if not isinstance(raw_prompt, (list, tuple)) or len(raw_prompt) != 2:
                raise ValueError("native-tool raw_prompt must be exactly system+user")
            messages = [dict(_mapping(message, "raw_prompt message")) for message in raw_prompt]
            if [message.get("role") for message in messages] != ["system", "user"]:
                raise ValueError("native-tool raw_prompt must be exactly system+user")
            prompt_text = "\n".join(str(message.get("content", "")) for message in messages)
            if source.ground_truth in prompt_text:
                raise ValueError("native-tool prompt must not expose canonical ground truth")

            output = await super().run(sampling_params, **kwargs)
            final_ids = final_policy_token_ids(output.response_ids, output.response_mask)
            final_text = (
                self.tokenizer.decode(list(final_ids), skip_special_tokens=True).strip()
                if final_ids
                else ""
            )
            try:
                parse_final_tool_answer(
                    final_text, registry=self.registry, grading=grading
                )
            except (TypeError, ValueError):
                terminal_valid = False
            else:
                terminal_valid = True
            output.reward_score = exact_tool_reward(
                final_text,
                ground_truth=source.ground_truth,
                ground_truth_level=source.ground_truth_level,
                registry=self.registry,
                grading=grading,
            )
            output.extra_fields.update(
                {
                    "trajectory_format": "qwen3.5-native-tools-v1",
                    "terminal_answer_valid": terminal_valid,
                    "terminal_exact_match": output.reward_score == 1.0,
                }
            )
            return output


__all__ = [
    "VERL_AGENT_LOOP_AVAILABLE",
    "DataClassifyCascadeAgentLoop",
    "final_policy_token_ids",
]
