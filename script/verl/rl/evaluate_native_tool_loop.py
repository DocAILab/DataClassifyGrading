"""Inference-only native-tool rollout gate for a merged HF checkpoint.

The evaluator deliberately reuses the repository's production
``DataClassifyCascadeAgentLoop`` and VeRL's ``qwen3_coder`` parser.  It only
supplies a local token-in/token-out vLLM adapter; it never constructs an
optimizer or starts a trainer.
"""

from __future__ import annotations

import argparse
import asyncio
from contextvars import ContextVar
from dataclasses import dataclass, field
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence
import uuid

from agent.release_policy import FORMAL_RELEASE_FORMAT, FORMAL_RELEASE_NAME
from agent.task import LeafRegistry
from agent.task.assets import load_corpus_categories
from agent.task.grading_manifest import DatasetGradingManifest
from agent.training.rl.sample import NATIVE_TOOL_TRAJECTORY_FORMAT, PROMPT_METADATA_FIELDS

_EXPECTED_TOOLS = frozenset(
    {
        "browse_categories",
        "get_category_details",
        "get_category_examples",
        "search_categories",
    }
)
_ALLOWED_EXTRA_INFO = frozenset(
    {
        "dataset",
        "stage",
        "source_id",
        "metadata",
        "ground_truth_level",
        "trajectory_format",
    }
)
_GATE_MIN = 0.05
_GATE_MAX = 0.30
_CURRENT_TRACKER: ContextVar["_EpisodeTracker | None"] = ContextVar(
    "native_rollout_episode_tracker", default=None
)


@dataclass(frozen=True)
class NativeEpisode:
    """One validated system+user episode accepted by the production loop."""

    source_id: str
    raw_prompt: tuple[dict[str, Any], dict[str, Any]]
    extra_info: dict[str, Any]
    reward_model: dict[str, Any]


@dataclass
class _EpisodeTracker:
    generation_calls: int = 0
    successful_tool_calls: int = 0


@dataclass
class _EpisodeResult:
    source_id: str
    reward: float = 0.0
    terminal_answer_valid: bool = False
    terminal_exact_match: bool = False
    num_turns: int = 0
    response_tokens: int = 0
    tool_calls: int = 0
    successful_tool_calls: int = 0
    tool_call_failures: int = 0
    generation_calls: int = 0
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "reward": self.reward,
            "terminal_answer_valid": self.terminal_answer_valid,
            "terminal_exact_match": self.terminal_exact_match,
            "num_turns": self.num_turns,
            "response_tokens": self.response_tokens,
            "tool_calls": self.tool_calls,
            "successful_tool_calls": self.successful_tool_calls,
            "tool_call_failures": self.tool_call_failures,
            "generation_calls": self.generation_calls,
            "errors": list(self.errors),
        }


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _normalize_prompt(value: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("native-tool raw_prompt must be exactly system+user")
    messages: list[dict[str, Any]] = []
    for message in value:
        if not isinstance(message, Mapping):
            raise ValueError("native-tool raw_prompt messages must be mappings")
        messages.append(dict(message))
    if [message.get("role") for message in messages] != ["system", "user"]:
        raise ValueError("native-tool raw_prompt must be exactly system+user")
    for message in messages:
        if not isinstance(message.get("content"), str):
            raise ValueError("native-tool raw_prompt content must be strings")
    return messages[0], messages[1]


def _normalize_extra_info(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("native-tool extra_info must be a mapping")
    extra = dict(value)
    unknown = set(extra) - _ALLOWED_EXTRA_INFO
    if unknown:
        raise ValueError(
            "formal native-tool extra_info contains unexpected keys: "
            + ", ".join(sorted(str(key) for key in unknown))
        )
    if extra.get("dataset") != FORMAL_RELEASE_NAME:
        raise ValueError(f"formal native-tool dataset must be {FORMAL_RELEASE_NAME}")
    if extra.get("stage") != "stage1":
        raise ValueError("formal native-tool stage must be stage1")
    _nonempty_string(extra.get("source_id"), "extra_info.source_id")
    _nonempty_string(extra.get("ground_truth_level"), "extra_info.ground_truth_level")
    if extra.get("trajectory_format") != NATIVE_TOOL_TRAJECTORY_FORMAT:
        raise ValueError(
            "formal native-tool trajectory_format must be "
            + NATIVE_TOOL_TRAJECTORY_FORMAT
        )
    metadata = extra.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("formal native-tool metadata must be a mapping")
    if set(metadata) != set(PROMPT_METADATA_FIELDS):
        raise ValueError(
            "formal native-tool metadata must be exactly "
            + "+".join(PROMPT_METADATA_FIELDS)
        )
    _nonempty_string(metadata.get("field_name"), "metadata.field_name")
    if not isinstance(metadata.get("table_name"), str):
        raise ValueError("metadata.table_name must be a string")
    return extra


def _assert_prompt_hides_ground_truth(
    raw_prompt: tuple[dict[str, Any], dict[str, Any]], reward_model: Mapping[str, Any]
) -> None:
    ground_truth = str(reward_model["ground_truth"])
    prompt_text = "\n".join(str(message.get("content", "")) for message in raw_prompt)
    if ground_truth in prompt_text:
        raise ValueError("native-tool prompt must not expose canonical ground truth")


def _normalize_reward_model(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("native-tool reward_model must be a mapping")
    reward_model = dict(value)
    if set(reward_model) != {"style", "ground_truth"}:
        raise ValueError("native-tool reward_model must be exactly style/ground_truth")
    if reward_model.get("style") != "rule":
        raise ValueError("native-tool reward_model.style must be rule")
    _nonempty_string(reward_model.get("ground_truth"), "reward_model.ground_truth")
    return reward_model


def normalize_episode_row(row: Mapping[str, Any]) -> NativeEpisode:
    """Normalize one RL five-field row or one trajectory SFT row.

    RL parquet is the preferred input.  The explicit trajectory compatibility
    form exists for the released 219-row post-SFT test artifact and still
    passes only the first system+user messages to the production loop.
    """

    if not isinstance(row, Mapping):
        raise ValueError("native-tool dataset row must be a mapping")
    if "prompt" in row:
        if row.get("data_source") != f"{FORMAL_RELEASE_NAME}/stage1":
            raise ValueError("native-tool data_source must be shougang/stage1")
        raw_prompt = _normalize_prompt(row.get("prompt"))
        extra = _normalize_extra_info(row.get("extra_info"))
        reward_model = _normalize_reward_model(row.get("reward_model"))
        _assert_prompt_hides_ground_truth(raw_prompt, reward_model)
        return NativeEpisode(
            source_id=str(extra["source_id"]),
            raw_prompt=raw_prompt,
            extra_info=extra,
            reward_model=reward_model,
        )

    if "messages" not in row:
        raise ValueError("native-tool row must contain prompt or messages")
    if row.get("stage") != "tool_trajectory":
        raise ValueError("trajectory compatibility rows must have stage tool_trajectory")
    if row.get("trajectory_format") != NATIVE_TOOL_TRAJECTORY_FORMAT:
        raise ValueError(
            "trajectory compatibility row has an unexpected trajectory_format"
        )
    source_id = _nonempty_string(row.get("source_id"), "source_id")
    metadata = row.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("trajectory metadata must be a mapping")
    extra = _normalize_extra_info(
        {
            "dataset": row.get("dataset", FORMAL_RELEASE_NAME),
            "stage": "stage1",
            "source_id": source_id,
            "metadata": dict(metadata),
            "ground_truth_level": row.get("ground_truth_level"),
            "trajectory_format": row.get("trajectory_format"),
        }
    )
    reward_model = _normalize_reward_model(
        {"style": "rule", "ground_truth": row.get("ground_truth")}
    )
    messages = row.get("messages")
    if not isinstance(messages, (list, tuple)) or len(messages) < 2:
        raise ValueError("trajectory row must contain system+user messages")
    raw_prompt = _normalize_prompt(messages[:2])
    _assert_prompt_hides_ground_truth(raw_prompt, reward_model)
    return NativeEpisode(
        source_id=source_id,
        raw_prompt=raw_prompt,
        extra_info=extra,
        reward_model=reward_model,
    )


def aggregate_gate(results: Sequence[Mapping[str, Any] | int | float | bool]) -> dict[str, Any]:
    """Compute the bounded non-zero reward gate with inclusive endpoints."""

    numerator = 0
    for item in results:
        if isinstance(item, Mapping):
            value = item.get("reward", 0.0)
        else:
            value = item
        try:
            numerator += int(float(value) > 0.0)
        except (TypeError, ValueError):
            raise ValueError("rollout reward values must be numeric") from None
    denominator = len(results)
    rate = numerator / denominator if denominator else 0.0
    return {
        "name": "post_sft_native_tool_rollout",
        "metric": "nonzero_reward_rate",
        "numerator": numerator,
        "denominator": denominator,
        "rate": rate,
        "min": _GATE_MIN,
        "max": _GATE_MAX,
        "inclusive": True,
        "passed": denominator > 0 and _GATE_MIN <= rate <= _GATE_MAX,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_tree(root: Path) -> str:
    if not root.is_dir():
        raise NotADirectoryError(root)
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: str | Path, *, tree: bool = False) -> dict[str, str]:
    value = Path(path).expanduser().resolve()
    if tree:
        return {"path": str(value), "sha256": _sha256_tree(value)}
    if not value.is_file():
        raise FileNotFoundError(value)
    return {"path": str(value), "sha256": _sha256_file(value)}


def _validate_release_report(
    report_path: Path,
    *,
    parquet_path: Path,
    parquet_sha256: str,
    rows: int,
    expected_report_sha256: str | None = None,
    expected_parquet_sha256: str | None = None,
) -> dict[str, Any]:
    """Verify the immutable release manifest before loading the model."""

    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("format") != "verl_tool_trajectory_messages_parquet":
        raise ValueError("release report has an unexpected format")
    if report.get("dataset") != FORMAL_RELEASE_NAME:
        raise ValueError("release report dataset is not shougang")
    if report.get("trajectory_format") != NATIVE_TOOL_TRAJECTORY_FORMAT:
        raise ValueError("release report trajectory_format does not match native tools")
    if report.get("metadata_fields") != list(PROMPT_METADATA_FIELDS):
        raise ValueError("release report metadata_fields do not match the formal contract")
    release = report.get("release")
    if not isinstance(release, Mapping) or release.get("status") != "passed" or not release.get("published"):
        raise ValueError("release report is not a published passed release")
    test_split = report.get("splits", {}).get("test")
    if not isinstance(test_split, Mapping) or not 0 < rows <= int(test_split.get("exported_records", -1)):
        raise ValueError("release report test row count does not cover evaluated parquet rows")
    expected_hash = test_split.get("parquet_sha256")
    if expected_hash is None:
        expected_hash = release.get("artifacts_sha256", {}).get(parquet_path.name)
    if expected_hash != parquet_sha256:
        raise ValueError("release report parquet_sha256 does not match parquet bytes")
    if expected_report_sha256 is not None and _sha256_file(report_path) != expected_report_sha256:
        raise ValueError("release report sha256 does not match the expected provenance hash")
    if expected_parquet_sha256 is not None and parquet_sha256 != expected_parquet_sha256:
        raise ValueError("parquet sha256 does not match the expected provenance hash")
    return report


def _validate_formal_assets(
    episodes: Sequence[NativeEpisode], *, registry_path: Path, corpus_path: Path, grading_manifest_path: Path
) -> None:
    """Fail closed on release/registry/rubric mismatches before model startup."""

    registry = LeafRegistry.from_path(registry_path)
    corpus = {
        item.category_id: item for item in load_corpus_categories(corpus_path)
    }
    if set(registry.ids) != set(corpus):
        raise ValueError("native-tool corpus must exactly cover the registry")
    manifest = DatasetGradingManifest.from_path(grading_manifest_path)
    grading = manifest.config_for(FORMAL_RELEASE_NAME)
    if grading.gt_field != "data_level":
        raise ValueError("formal shougang grading gt_field must be data_level")
    for episode in episodes:
        ground_truth = str(episode.reward_model["ground_truth"])
        level = str(episode.extra_info["ground_truth_level"])
        if ground_truth not in registry.ids:
            raise ValueError("native-tool ground_truth is absent from the registry")
        if level not in grading.levels:
            raise ValueError("native-tool ground_truth_level is outside the approved rubric")


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _build_config(
    model: Path,
    *,
    max_prompt_length: int,
    max_response_length: int,
    max_assistant_turns: int,
    max_user_turns: int,
    max_parallel_calls: int,
    max_tool_response_length: int,
    agent_format: str,
) -> Any:
    from omegaconf import OmegaConf

    return OmegaConf.create(
        {
            "actor_rollout_ref": {
                "model": {"path": str(model), "tokenizer_path": str(model)},
                "rollout": {
                    "name": "vllm",
                    "prompt_length": max_prompt_length,
                    "response_length": max_response_length,
                    "multi_turn": {
                        "format": agent_format,
                        "max_assistant_turns": max_assistant_turns,
                        "max_user_turns": max_user_turns,
                        "max_parallel_calls": max_parallel_calls,
                        "max_tool_response_length": max_tool_response_length,
                        "tool_response_truncate_side": "right",
                    },
                },
            },
            "data": {
                "continuous_token": {"enable": False},
                "apply_chat_template_kwargs": {"enable_thinking": False},
                "mm_processor_kwargs": {},
            },
        }
    )


class _AsyncVllmServer:
    """Adapt vLLM's async token stream to AgentLoop's token seam."""

    def __init__(self, engine: Any) -> None:
        self.engine = engine

    async def generate(
        self,
        request_id: str,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        **_: Any,
    ) -> Any:
        tracker = _CURRENT_TRACKER.get()
        if tracker is not None:
            tracker.generation_calls += 1
        from verl.workers.rollout.replica import TokenOutput
        from vllm import SamplingParams, TokensPrompt

        params = SamplingParams(**sampling_params)
        request = TokensPrompt(prompt_token_ids=list(prompt_ids))
        final_output = None
        async for output in self.engine.generate(
            request, params, request_id=request_id
        ):
            final_output = output
        if final_output is None or not final_output.outputs:
            raise RuntimeError("vLLM returned no completion output")
        completion = final_output.outputs[0]
        return TokenOutput(
            token_ids=list(completion.token_ids),
            log_probs=None,
            stop_reason=completion.finish_reason,
            num_preempted=0,
            extra_fields={},
        )


def _tracking_tools(tools: Sequence[Any]) -> list[Any]:
    from verl.tools.function_tool import FunctionTool

    tracked: list[FunctionTool] = []
    for original in tools:
        if original.is_async:

            async def tracked_async(*args: Any, _original=original.fn, **kwargs: Any) -> Any:
                tracker = _CURRENT_TRACKER.get()
                try:
                    result = await _original(*args, **kwargs)
                except Exception:
                    raise
                else:
                    if tracker is not None:
                        tracker.successful_tool_calls += 1
                    return result

            function = tracked_async
        else:

            def tracked_sync(*args: Any, _original=original.fn, **kwargs: Any) -> Any:
                tracker = _CURRENT_TRACKER.get()
                try:
                    result = _original(*args, **kwargs)
                except Exception:
                    raise
                else:
                    if tracker is not None:
                        tracker.successful_tool_calls += 1
                    return result

            function = tracked_sync
        tracked.append(
            FunctionTool(
                name=original.name,
                fn=function,
                tool_schema=original.tool_schema,
                is_async=original.is_async,
            )
        )
    return tracked


async def _run_episodes(
    episodes: Sequence[NativeEpisode],
    *,
    model: Path,
    tools_path: Path,
    registry_path: Path,
    corpus_path: Path,
    grading_manifest_path: Path,
    seed: int,
    max_prompt_length: int,
    max_response_length: int,
    max_assistant_turns: int,
    max_user_turns: int,
    max_parallel_calls: int,
    max_tool_response_length: int,
    max_model_len: int,
    gpu_memory_utilization: float,
    enforce_eager: bool,
) -> list[dict[str, Any]]:
    from transformers import AutoProcessor, AutoTokenizer
    from verl.experimental.agent_loop.agent_loop import DictConfigWrap, ToolListWrap
    from verl.tools.function_tool import load_function_tools_from_path
    from verl.utils.dataset.rl_dataset import RLHFDataset
    from script.verl.rl.cascade_agent_loop import DataClassifyCascadeAgentLoop
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    # Qwen3.5 on Blackwell trips vLLM's FlashInfer sampler capability probe;
    # keep this evaluator on the known-good native sampler path.
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    os.environ["DATACLASSIFY_RLOO_REGISTRY"] = str(registry_path)
    os.environ["DATACLASSIFY_RLOO_CORPUS"] = str(corpus_path)
    os.environ["DATACLASSIFY_RLOO_GRADING_MANIFEST"] = str(grading_manifest_path)
    tools = load_function_tools_from_path(str(tools_path))
    if {tool.name for tool in tools} != _EXPECTED_TOOLS:
        raise ValueError(
            "native-tool function path must define exactly: "
            + ", ".join(sorted(_EXPECTED_TOOLS))
        )
    tracked_tools = _tracking_tools(tools)
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    processor = AutoProcessor.from_pretrained(model, local_files_only=True)
    engine_args = AsyncEngineArgs(
        model=str(model),
        tokenizer=str(model),
        runner="generate",
        dtype="bfloat16",
        seed=seed,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        enforce_eager=enforce_eager,
    )
    # AsyncLLM must be created and consumed from this running event loop;
    # constructing it outside asyncio.run can strand EngineCore's loop.
    engine = AsyncLLM.from_engine_args(engine_args)
    config = _build_config(
        model,
        max_prompt_length=max_prompt_length,
        max_response_length=max_response_length,
        max_assistant_turns=max_assistant_turns,
        max_user_turns=max_user_turns,
        max_parallel_calls=max_parallel_calls,
        max_tool_response_length=max_tool_response_length,
        agent_format="qwen3_coder",
    )
    try:
        agent_loop = DataClassifyCascadeAgentLoop(
            trainer_config=DictConfigWrap(config),
            server_manager=_AsyncVllmServer(engine),
            tokenizer=tokenizer,
            processor=processor,
            dataset_cls=RLHFDataset,
            data_config=DictConfigWrap(config.data),
            tools=ToolListWrap(tracked_tools),
        )
    except BaseException:
        engine.shutdown()
        raise
    sampling_params = {
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": -1,
        "max_tokens": max_response_length,
        "stop_token_ids": [tokenizer.eos_token_id],
        "skip_special_tokens": False,
    }
    results: list[dict[str, Any]] = []
    for index, episode in enumerate(episodes):
        tracker = _EpisodeTracker()
        token = _CURRENT_TRACKER.set(tracker)
        result = _EpisodeResult(source_id=episode.source_id)
        try:
            output = await agent_loop.run(
                dict(sampling_params),
                raw_prompt=[dict(message) for message in episode.raw_prompt],
                extra_info=episode.extra_info,
                reward_model=episode.reward_model,
            )
            result.reward = float(output.reward_score)
            result.terminal_answer_valid = bool(
                output.extra_fields.get("terminal_answer_valid", False)
            )
            result.terminal_exact_match = bool(
                output.extra_fields.get("terminal_exact_match", False)
            )
            result.num_turns = int(output.num_turns)
            result.response_tokens = len(output.response_ids)
            rewards = output.extra_fields.get("tool_rewards", [])
            result.tool_calls = len(rewards) if isinstance(rewards, list) else 0
            result.successful_tool_calls = tracker.successful_tool_calls
            result.tool_call_failures = max(
                0, result.tool_calls - result.successful_tool_calls
            )
            result.generation_calls = tracker.generation_calls
        except Exception as exc:  # a failed episode counts as zero, never as a pass
            result.errors.append(f"{type(exc).__name__}: {exc}")
            result.generation_calls = tracker.generation_calls
        finally:
            _CURRENT_TRACKER.reset(token)
        results.append(result.as_dict())
        print(
            json.dumps(
                {
                    "index": index,
                    "source_id": result.source_id,
                    "reward": result.reward,
                    "terminal_answer_valid": result.terminal_answer_valid,
                    "tool_calls": result.tool_calls,
                    "errors": result.errors,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    engine.shutdown()
    return results


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--base-model", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--merge-report", type=Path)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--release-dir", type=Path)
    parser.add_argument("--release-report", type=Path, required=True)
    parser.add_argument("--expected-release-report-sha256")
    parser.add_argument("--expected-parquet-sha256")
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--grading-manifest", type=Path, required=True)
    parser.add_argument("--tools", type=Path, required=True)
    parser.add_argument("--agent-loop-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-prompt-length", type=int, default=7168)
    parser.add_argument("--max-response-length", type=int, default=2048)
    parser.add_argument("--max-assistant-turns", type=int, default=4)
    parser.add_argument("--max-user-turns", type=int, default=3)
    parser.add_argument("--max-parallel-calls", type=int, default=1)
    parser.add_argument("--max-tool-response-length", type=int, default=4096)
    parser.add_argument("--max-model-len", type=int, default=12288)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument(
        "--enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def _validate_options(args: argparse.Namespace) -> None:
    for name in (
        "model",
        "data",
        "release_report",
        "registry",
        "corpus",
        "grading_manifest",
        "tools",
        "agent_loop_config",
    ):
        if not getattr(args, name).exists():
            raise FileNotFoundError(getattr(args, name))
    if args.base_model is not None and not args.base_model.is_dir():
        raise FileNotFoundError(args.base_model)
    if args.checkpoint is not None and not args.checkpoint.is_dir():
        raise FileNotFoundError(args.checkpoint)
    if args.merge_report is not None and not args.merge_report.is_file():
        raise FileNotFoundError(args.merge_report)
    if args.limit == 0 or args.limit < -1:
        raise ValueError("limit must be -1 or positive")
    for name in (
        "max_prompt_length",
        "max_response_length",
        "max_assistant_turns",
        "max_user_turns",
        "max_parallel_calls",
        "max_tool_response_length",
        "max_model_len",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if args.max_parallel_calls != 1:
        raise ValueError("formal native-tool max_parallel_calls must be 1")
    if args.max_model_len < args.max_prompt_length + args.max_response_length:
        raise ValueError("max_model_len must cover prompt plus response length")
    if not 0.0 < args.gpu_memory_utilization < 1.0:
        raise ValueError("gpu_memory_utilization must be within (0, 1)")


def _write_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite rollout report: {path}")
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        handle.write(text)
        staging = Path(handle.name)
    try:
        staging.replace(path)
    finally:
        staging.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        _validate_options(args)
        import pyarrow.parquet as pq

        rows = pq.read_table(args.data).to_pylist()
        if args.limit > 0:
            rows = rows[: args.limit]
        episodes = [normalize_episode_row(row) for row in rows]
        if not episodes:
            raise ValueError("native-tool dataset has no episodes")
        # Bind all immutable inputs before the model is started.
        data_artifact = _artifact(args.data)
        release_artifact = _artifact(args.release_report)
        _validate_release_report(
            args.release_report.expanduser().resolve(),
            parquet_path=args.data.expanduser().resolve(),
            parquet_sha256=data_artifact["sha256"],
            rows=len(episodes),
            expected_report_sha256=args.expected_release_report_sha256,
            expected_parquet_sha256=args.expected_parquet_sha256,
        )
        _validate_formal_assets(
            episodes,
            registry_path=args.registry.expanduser().resolve(),
            corpus_path=args.corpus.expanduser().resolve(),
            grading_manifest_path=args.grading_manifest.expanduser().resolve(),
        )
        registry_artifact = _artifact(args.registry)
        corpus_artifact = _artifact(args.corpus)
        grading_artifact = _artifact(args.grading_manifest)
        tools_artifact = _artifact(args.tools)
        config_artifact = _artifact(args.agent_loop_config)
        model_artifact = _artifact(args.model, tree=True)
        base_model_artifact = (
            _artifact(args.base_model, tree=True) if args.base_model is not None else None
        )
        checkpoint_artifact = (
            _artifact(args.checkpoint, tree=True) if args.checkpoint is not None else None
        )
        merge_artifact = (
            _artifact(args.merge_report) if args.merge_report is not None else None
        )
        started = asyncio.run(
            _run_episodes(
                episodes,
                model=args.model.expanduser().resolve(),
                tools_path=args.tools.expanduser().resolve(),
                registry_path=args.registry.expanduser().resolve(),
                corpus_path=args.corpus.expanduser().resolve(),
                grading_manifest_path=args.grading_manifest.expanduser().resolve(),
                seed=args.seed,
                max_prompt_length=args.max_prompt_length,
                max_response_length=args.max_response_length,
                max_assistant_turns=args.max_assistant_turns,
                max_user_turns=args.max_user_turns,
                max_parallel_calls=args.max_parallel_calls,
                max_tool_response_length=args.max_tool_response_length,
                max_model_len=args.max_model_len,
                gpu_memory_utilization=args.gpu_memory_utilization,
                enforce_eager=args.enforce_eager,
            )
        )
        gate = aggregate_gate(started)
        errors = [
            {"source_id": result["source_id"], "errors": result["errors"]}
            for result in started
            if result["errors"]
        ]
        gate["passed"] = bool(gate["passed"] and not errors)
        n = len(started)
        tool_calls = sum(int(result["tool_calls"]) for result in started)
        successful_tool_calls = sum(
            int(result["successful_tool_calls"]) for result in started
        )
        counts = {
            "episodes": n,
            "nonzero_reward": gate["numerator"],
            "zero_reward": n - gate["numerator"],
            "terminal_valid": sum(
                bool(result["terminal_answer_valid"]) for result in started
            ),
            "exact_match": sum(
                bool(result["terminal_exact_match"]) for result in started
            ),
            "tool_call_episodes": sum(bool(result["tool_calls"]) for result in started),
            "tool_calls": tool_calls,
            "successful_tool_calls": successful_tool_calls,
            "tool_call_failures": sum(
                int(result["tool_call_failures"]) for result in started
            ),
            "generation_calls": sum(
                int(result["generation_calls"]) for result in started
            ),
            "episode_failures": len(errors),
        }
        runtime: dict[str, Any] = {
            "python": sys.executable,
            "verl": _version("verl"),
            "vllm": _version("vllm"),
            "transformers": _version("transformers"),
            "torch": _version("torch"),
        }
        try:
            import torch

            runtime["cuda"] = torch.version.cuda
            runtime["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        except Exception as exc:
            runtime["gpu_probe_error"] = f"{type(exc).__name__}: {exc}"
        report = {
            "format": "dataclassify-native-tool-rollout-gate-v1",
            "status": "passed" if gate["passed"] else "failed",
            "accepted": bool(gate["passed"]),
            "gate": gate,
            "dataset": {
                "name": FORMAL_RELEASE_NAME,
                "release_format": FORMAL_RELEASE_FORMAT,
                "release_dir": str(
                    (args.release_dir or args.data.parent).expanduser().resolve()
                ),
                "report_path": release_artifact["path"],
                "report_sha256": release_artifact["sha256"],
                "parquet_path": data_artifact["path"],
                "parquet_sha256": data_artifact["sha256"],
                "split": "test",
                "rows": n,
                "trajectory_format": NATIVE_TOOL_TRAJECTORY_FORMAT,
                "metadata_fields": list(PROMPT_METADATA_FIELDS),
                "registry_path": registry_artifact["path"],
                "registry_sha256": registry_artifact["sha256"],
                "corpus_path": corpus_artifact["path"],
                "corpus_sha256": corpus_artifact["sha256"],
                "grading_manifest_path": grading_artifact["path"],
                "grading_manifest_sha256": grading_artifact["sha256"],
                "registry": registry_artifact,
                "corpus": corpus_artifact,
                "grading_manifest": grading_artifact,
            },
            "model": {
                "path": model_artifact["path"],
                "artifact_kind": "merged_hf",
                "tree_sha256": model_artifact["sha256"],
                "config_sha256": _sha256_file(args.model.expanduser().resolve() / "config.json"),
                "base_model": base_model_artifact,
                "checkpoint": checkpoint_artifact,
                "merge_report": merge_artifact,
            },
            "inference": {
                "backend": "vllm.v1.engine.async_llm.AsyncLLM via local token-in/token-out adapter",
                "model_load_class": "AutoProcessor + AutoTokenizer; vLLM Qwen3_5ForConditionalGeneration",
                "sampling": {
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "top_k": -1,
                    "max_tokens": args.max_response_length,
                    "request_seed": None,
                    "engine_seed": args.seed,
                },
                "max_prompt_length": args.max_prompt_length,
                "max_response_length": args.max_response_length,
                "max_model_len": args.max_model_len,
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "enforce_eager": args.enforce_eager,
                "flashinfer_sampler": "disabled by VLLM_USE_FLASHINFER_SAMPLER=0",
                "runtime": runtime,
            },
            "parser": {
                "tool_parser": "qwen3_coder",
                "terminal_parser": "agent.training.rl.native_tools.parse_final_tool_answer",
                "reward_fn": "agent.training.rl.native_tools.exact_tool_reward",
                "terminal_segment_rule": "trailing_contiguous_policy_mask",
                "strict_keys": ["answer", "level"],
            },
            "agent_loop": {
                "name": "dataclassify_cascade",
                "class": "script.verl.rl.cascade_agent_loop.DataClassifyCascadeAgentLoop",
                "config": config_artifact,
                "function_tool_path": tools_artifact,
                "tools": sorted(_EXPECTED_TOOLS),
                "max_assistant_turns": args.max_assistant_turns,
                "max_user_turns": args.max_user_turns,
                "max_parallel_calls": args.max_parallel_calls,
                "max_tool_response_length": args.max_tool_response_length,
                "truncate_side": "right",
                "tokenization_sanity_check_mode": "strict",
                "response_length": args.max_response_length,
            },
            "counts": counts,
            "per_source": started,
            "errors": errors,
        }
        _write_atomic(args.output.expanduser().resolve(), report)
        print(json.dumps({"report": str(args.output.resolve()), "gate": gate, "counts": counts}, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError, TypeError) as exc:
        print(f"native_tool_loop: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


__all__ = ["NativeEpisode", "aggregate_gate", "normalize_episode_row", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
