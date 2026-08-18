"""Unified RL data, parser and reward contracts for the two-stage task.

Stage 4A scope: the reusable interfaces future RL training consumes —
RL samples (resolved-only, target.category_id as the only label), the VeRL
v0.8.0 five-field RL parquet exporter/validator, the unified output parser
(agent.task.parser.check_*), and the shared task reward. No training
algorithm, rollout engine, vLLM/Ray or GPU code lives here.
"""

from .dataset import (
    RL_SPLITS,
    VERL_RL_COLUMNS,
    export_rl_dataset,
    validate_rl_dataset,
)
from .reward import (
    FULL_REWARD,
    INVALID_REWARD,
    STAGE1_VALID_MISS_DEFAULT,
    STAGE2_PARTIAL_DEFAULT,
    RewardConfig,
    RewardResult,
    reward_for_parse_result,
    reward_stage1,
    reward_stage2,
)
from .sample import (
    RewardMeta,
    RlMessage,
    RlSample,
    build_rl_row,
    build_rl_samples,
    visible_metadata,
)

__all__ = [
    "RlMessage",
    "RewardMeta",
    "RlSample",
    "visible_metadata",
    "build_rl_samples",
    "build_rl_row",
    "RewardConfig",
    "RewardResult",
    "FULL_REWARD",
    "INVALID_REWARD",
    "STAGE1_VALID_MISS_DEFAULT",
    "STAGE2_PARTIAL_DEFAULT",
    "reward_stage1",
    "reward_stage2",
    "reward_for_parse_result",
    "RL_SPLITS",
    "VERL_RL_COLUMNS",
    "export_rl_dataset",
    "validate_rl_dataset",
]
