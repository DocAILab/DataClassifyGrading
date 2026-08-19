"""Thin VeRL reward adapter (choice protocol, Phase 12).

Routes VeRL's ``compute_score(data_source, solution_str, ground_truth,
extra_info, **kwargs)`` to the shared choice-aware task reward
(``reward_stage1_choices`` / ``reward_stage2_choices``). This module
implements NO parser, NO correctness logic and NO reward table of its own —
it only converts the VeRL calling convention and routes by
``<dataset>/stage<1|2>``:

- ``data_source`` == ``"<dataset>/stage1"`` -> ``reward_stage1_choices``
- ``data_source`` == ``"<dataset>/stage2"`` -> ``reward_stage2_choices``
  (Stage 2 candidates come from ``extra_info["candidates"]``, written by the
  RL exporter and carried through VeRL's non-tensor batch).

The choice-aware reward decodes the model's choice ids to canonical category
ids BEFORE the unchanged canonical reward table applies, so
``choice-id output -> shared choice parser -> canonical decode -> reward``
holds end to end. The LeafRegistry required by the shared reward is loaded
once per dataset from ``DATACLASSIFY_REGISTRY_DIR`` (default
``cfg/task/registry``) and cached. A test hook (``configure()``) lets
callers pre-register explicit registries / reward config for unit tests.

Observation hook (smoke only, off by default): when the environment variable
``DATACLASSIFY_REWARD_LOG_DIR`` is set, every reward call appends one JSON
line (data_source, solution, reward, reason, parse validity, decoded
canonical prediction) to ``<dir>/reward_runtime.jsonl``. This changes no
reward value — it only makes the real loop observable for the Phase 12
vertical slice.

GRPO alone does not leak into this module: everything here is
algorithm-agnostic reward routing on top of the shared task layer.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Mapping

from agent.task.contracts import LeafRegistry
from agent.training.rl.reward import (
    RewardConfig,
    RewardResult,
    reward_stage1_choices,
    reward_stage2_choices,
)

DEFAULT_REGISTRY_DIR = "cfg/task/registry"


def parse_data_source(data_source: str) -> tuple[str, str]:
    """Split ``"<dataset>/stage1|stage2"`` into (dataset, stage)."""
    if not isinstance(data_source, str):
        raise ValueError(f"data_source must be a string, got {type(data_source).__name__}")
    try:
        dataset, stage = data_source.rsplit("/", 1)
    except ValueError:
        raise ValueError(f"data_source must be '<dataset>/stage<1|2>', got {data_source!r}") from None
    if stage not in {"stage1", "stage2"}:
        raise ValueError(f"data_source stage must be stage1 or stage2, got {stage!r}")
    if not dataset.strip():
        raise ValueError("data_source dataset must be non-empty")
    return dataset, stage


def _decoded_entries(result: RewardResult) -> list[str] | None:
    """Canonical prediction carried by a choice reward result (if decodable)."""
    output = result.parsed_output
    if output is None:
        return None
    if hasattr(output, "candidates") and output.candidates:
        return list(output.candidates)
    if hasattr(output, "answer") and output.answer:
        return [output.answer]
    return None


def _log_observation(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    result: RewardResult,
) -> None:
    log_dir = os.environ.get("DATACLASSIFY_REWARD_LOG_DIR")
    if not log_dir:
        return
    record = {
        "data_source": data_source,
        "solution": solution_str,
        "ground_truth": ground_truth,
        "reward": result.reward,
        "reason": result.reason,
        "format_valid": result.format_valid,
        "constraint_valid": result.constraint_valid,
        "task_correct": result.task_correct,
        "decoded_canonical": _decoded_entries(result),
    }
    platform_log_dir = Path(log_dir)
    platform_log_dir.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False)
    with (platform_log_dir / "reward_runtime.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


class _Router:
    """Holds per-dataset registries + reward config; routes one reward call."""

    def __init__(self) -> None:
        self._registries: dict[str, LeafRegistry] = {}
        self._config: RewardConfig = RewardConfig()
        self._registry_dir: str | None = None
        self._lock = threading.Lock()

    def set_registries(self, registries: Mapping[str, LeafRegistry]) -> None:
        with self._lock:
            self._registries.update(dict(registries))

    def set_registry_dir(self, directory: str | Path | None) -> None:
        with self._lock:
            self._registry_dir = None if directory is None else str(directory)

    def set_reward_config(self, config: RewardConfig) -> None:
        with self._lock:
            self._config = config

    def registry_for(self, dataset: str) -> LeafRegistry:
        registry = self._registries.get(dataset)
        if registry is not None:
            return registry
        with self._lock:
            registry = self._registries.get(dataset)
            if registry is None:
                directory = self._registry_dir or os.environ.get(
                    "DATACLASSIFY_REGISTRY_DIR", DEFAULT_REGISTRY_DIR
                )
                path = Path(directory) / f"{dataset}.registry.json"
                registry = LeafRegistry.from_path(path)
                self._registries[dataset] = registry
        return registry

    def __call__(
        self,
        data_source: str,
        solution_str: str,
        ground_truth: str,
        extra_info: Mapping | None,
        **kwargs: object,
    ) -> float:
        dataset, stage = parse_data_source(data_source)
        registry = self.registry_for(dataset)
        info: Mapping = extra_info if isinstance(extra_info, Mapping) else {}
        if stage == "stage1":
            result = reward_stage1_choices(
                solution_str,
                ground_truth=ground_truth,
                registry=registry,
                config=self._config,
            )
        else:
            candidates = info.get("candidates")
            result = reward_stage2_choices(
                solution_str,
                ground_truth=ground_truth,
                candidates=candidates,
                registry=registry,
                config=self._config,
            )
        _log_observation(data_source, solution_str, ground_truth, result)
        return float(result.reward)


_ROUTER = _Router()


def configure(
    *,
    registries: Mapping[str, LeafRegistry] | None = None,
    registry_dir: str | Path | None = None,
    reward_config: RewardConfig | None = None,
) -> None:
    """Test/embedding hook: pre-register registries and reward config.

    Production uses ``DATACLASSIFY_REGISTRY_DIR`` (lazy per-dataset load),
    which is unaffected unless explicitly overridden here.
    """
    if registries is not None:
        _ROUTER.set_registries(registries)
    if registry_dir is not None:
        _ROUTER.set_registry_dir(registry_dir)
    if reward_config is not None:
        _ROUTER.set_reward_config(reward_config)


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Mapping | None,
    **kwargs: object,
) -> float:
    """VeRL reward adapter entrypoint (matches verl reward-manager signature).

    Never raises on illegal model output (the shared reward maps it to 0.0);
    a malformed ``data_source`` is a programming error and raises.
    """
    return _ROUTER(data_source, solution_str, ground_truth, extra_info, **kwargs)


__all__ = ["compute_score", "configure", "parse_data_source", "DEFAULT_REGISTRY_DIR"]
