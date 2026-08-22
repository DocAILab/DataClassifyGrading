"""Standalone VeRL v0.8.0 RLOO experiment adapter.

This file is both the VeRL custom-reward entrypoint (``compute_score``) and a
launcher that selects RLOO through Hydra overrides. It deliberately keeps all
algorithm configuration outside ``agent.task`` and ``agent.training``.

The launcher requires explicit runtime-local model/data/registry/corpus/task
paths. Use ``--dry-run`` locally; GPU execution must wait for an authorized
server validation session.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from functools import lru_cache
from importlib import metadata as importlib_metadata
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

from agent.task import LeafRegistry, TaskConfig
from agent.training.rl.reward import reward_stage1_choices, reward_stage2_choices

_EXPECTED_VERL_VERSION = "0.8.0"
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DATASET_ENV = "DATACLASSIFY_RLOO_DATASET"
_REGISTRY_ENV = "DATACLASSIFY_RLOO_REGISTRY"


@dataclass(frozen=True)
class RlooExperimentConfig:
    """All task paths and RLOO smoke knobs required by the adapter."""

    dataset: str
    model_path: Path
    data_dir: Path
    registry_path: Path
    corpus_path: Path
    task_config_path: Path
    output_dir: Path
    python_bin: str = sys.executable
    rollout_n: int = 2
    train_batch_size: int = 4
    ppo_mini_batch_size: int = 4
    ppo_micro_batch_size: int = 1
    train_max_samples: int = 8
    val_max_samples: int = 2
    max_prompt_length: int = 1024
    max_response_length: int = 256
    max_model_len: int = 2048
    ppo_max_token_len_per_gpu: int = 16384
    total_training_steps: int = 3
    actor_lr: float = 1e-4
    kl_coef: float = 0.001
    gpu_memory_utilization: float = 0.5
    lora_rank: int = 8
    lora_alpha: int = 16
    enforce_eager: bool = False
    param_offload: bool = False
    save_freq: int = 1000
    test_freq: int = 1000

    @property
    def train_file(self) -> Path:
        return self.data_dir / "train.parquet"

    @property
    def val_file(self) -> Path:
        return self.data_dir / "val.parquet"

    @property
    def test_file(self) -> Path:
        return self.data_dir / "test.parquet"

    def validate_options(self) -> None:
        if not self.dataset.strip():
            raise ValueError("dataset must be non-empty")
        if not self.python_bin.strip():
            raise ValueError("python_bin must be non-empty")
        if self.rollout_n < 2:
            raise ValueError("RLOO rollout_n must be at least 2")
        positive_integers = {
            "train_batch_size": self.train_batch_size,
            "ppo_mini_batch_size": self.ppo_mini_batch_size,
            "ppo_micro_batch_size": self.ppo_micro_batch_size,
            "max_prompt_length": self.max_prompt_length,
            "max_response_length": self.max_response_length,
            "max_model_len": self.max_model_len,
            "ppo_max_token_len_per_gpu": self.ppo_max_token_len_per_gpu,
            "total_training_steps": self.total_training_steps,
        }
        for name, value in positive_integers.items():
            if value < 1:
                raise ValueError(f"{name} must be positive")
        for name, value in (
            ("train_max_samples", self.train_max_samples),
            ("val_max_samples", self.val_max_samples),
        ):
            if value != -1 and value < 1:
                raise ValueError(f"{name} must be -1 or positive")
        if self.train_batch_size % self.ppo_mini_batch_size != 0:
            raise ValueError("train_batch_size must be divisible by ppo_mini_batch_size")
        if self.ppo_mini_batch_size % self.ppo_micro_batch_size != 0:
            raise ValueError("ppo_mini_batch_size must be divisible by ppo_micro_batch_size")
        sequence_length = self.max_prompt_length + self.max_response_length
        if self.max_model_len < sequence_length:
            raise ValueError("max_model_len must cover prompt plus response length")
        if self.ppo_max_token_len_per_gpu < sequence_length:
            raise ValueError("ppo_max_token_len_per_gpu must cover one full sequence")
        if not 0.0 < self.gpu_memory_utilization < 1.0:
            raise ValueError("gpu_memory_utilization must be within (0, 1)")
        if self.actor_lr <= 0:
            raise ValueError("actor_lr must be positive")
        if self.kl_coef < 0:
            raise ValueError("kl_coef must be non-negative")
        if self.lora_rank < 0 or self.lora_alpha < 1:
            raise ValueError("lora_rank must be non-negative and lora_alpha positive")

    def validate_local_paths(self) -> None:
        required_files = (
            self.train_file,
            self.val_file,
            self.test_file,
            self.registry_path,
            self.corpus_path,
            self.task_config_path,
        )
        for path in required_files:
            if not path.is_file():
                raise FileNotFoundError(f"required local file not found: {path}")
        if not (self.model_path / "config.json").is_file():
            raise FileNotFoundError(f"model config not found: {self.model_path / 'config.json'}")


def _bool(value: bool) -> str:
    return "True" if value else "False"


def build_validation_command(config: RlooExperimentConfig) -> list[str]:
    """Reuse the existing five-field RL contract validator unchanged."""
    config.validate_options()
    task = TaskConfig.from_path(config.task_config_path)
    return [
        config.python_bin,
        "-m",
        "script.verl.rl.validate",
        "--dataset-dir",
        str(config.data_dir),
        "--dataset",
        config.dataset,
        "--registry",
        str(config.registry_path),
        "--corpus",
        str(config.corpus_path),
        "--task-config",
        str(config.task_config_path),
        "--metadata-fields",
        *task.metadata_fields,
    ]


def build_verl_command(config: RlooExperimentConfig) -> list[str]:
    """Build the VeRL command; all RLOO math remains inside VeRL."""
    config.validate_options()
    reward_adapter = Path(__file__).resolve()
    overrides = [
        "algorithm.adv_estimator=rloo",
        "algorithm.use_kl_in_reward=True",
        "algorithm.kl_penalty=kl",
        f"algorithm.kl_ctrl.kl_coef={config.kl_coef}",
        f"data.train_files={config.train_file}",
        f"data.val_files={config.val_file}",
        f"data.train_batch_size={config.train_batch_size}",
        f"data.train_max_samples={config.train_max_samples}",
        f"data.val_max_samples={config.val_max_samples}",
        f"data.max_prompt_length={config.max_prompt_length}",
        f"data.max_response_length={config.max_response_length}",
        "data.shuffle=False",
        "data.truncation=error",
        "data.dataloader_num_workers=0",
        "data.prompt_key=prompt",
        "data.reward_fn_key=data_source",
        f"actor_rollout_ref.model.path={config.model_path}",
        "actor_rollout_ref.model.use_remove_padding=False",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        "+actor_rollout_ref.model.override_config.attn_implementation=sdpa",
        f"actor_rollout_ref.model.lora_rank={config.lora_rank}",
        f"actor_rollout_ref.model.lora_alpha={config.lora_alpha}",
        "actor_rollout_ref.model.target_modules=all-linear",
        "actor_rollout_ref.actor.strategy=fsdp",
        "actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False",
        f"actor_rollout_ref.actor.fsdp_config.param_offload={_bool(config.param_offload)}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={config.ppo_mini_batch_size}",
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={config.ppo_micro_batch_size}",
        f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu={config.ppo_max_token_len_per_gpu}",
        "actor_rollout_ref.actor.use_kl_loss=False",
        "actor_rollout_ref.actor.entropy_coeff=0",
        f"actor_rollout_ref.actor.optim.lr={config.actor_lr}",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        f"actor_rollout_ref.rollout.n={config.rollout_n}",
        f"actor_rollout_ref.rollout.prompt_length={config.max_prompt_length}",
        f"actor_rollout_ref.rollout.response_length={config.max_response_length}",
        f"actor_rollout_ref.rollout.max_model_len={config.max_model_len}",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={config.gpu_memory_utilization}",
        f"actor_rollout_ref.rollout.enforce_eager={_bool(config.enforce_eager)}",
        "actor_rollout_ref.rollout.enable_chunked_prefill=False",
        "actor_rollout_ref.rollout.enable_prefix_caching=False",
        "actor_rollout_ref.rollout.free_cache_engine=True",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.ref.fsdp_config.param_offload=True",
        f"reward.custom_reward_function.path={reward_adapter}",
        "reward.custom_reward_function.name=compute_score",
        "reward.reward_manager.name=naive",
        "reward.reward_model.enable=False",
        "reward.num_workers=1",
        "trainer.critic_warmup=0",
        "trainer.n_gpus_per_node=1",
        "trainer.nnodes=1",
        "trainer.logger=[\"console\"]",
        "trainer.project_name=dataclassify-rloo",
        f"trainer.experiment_name=rloo-smoke-{config.dataset}",
        f"trainer.default_local_dir={config.output_dir / 'checkpoints'}",
        "trainer.total_epochs=4",
        f"trainer.total_training_steps={config.total_training_steps}",
        f"trainer.save_freq={config.save_freq}",
        f"trainer.test_freq={config.test_freq}",
        "trainer.val_before_train=False",
        "trainer.resume_mode=disable",
    ]
    return [config.python_bin, "-m", "verl.trainer.main_ppo", *overrides]


@lru_cache(maxsize=8)
def _load_registry(path: str) -> LeafRegistry:
    return LeafRegistry.from_path(path)


def _route(data_source: str, extra_info: Mapping[str, Any] | None) -> tuple[str, Mapping[str, Any]]:
    if not isinstance(data_source, str):
        raise ValueError("data_source must be a string")
    try:
        dataset, stage = data_source.rsplit("/", 1)
    except ValueError:
        raise ValueError("data_source must be '<dataset>/stage1|stage2'") from None
    if stage not in {"stage1", "stage2"} or not dataset:
        raise ValueError("data_source must be '<dataset>/stage1|stage2'")
    expected_dataset = os.environ.get(_DATASET_ENV, "").strip()
    if not expected_dataset:
        raise RuntimeError(f"{_DATASET_ENV} is required")
    if dataset != expected_dataset:
        raise ValueError(
            f"data_source dataset {dataset!r} does not match configured dataset "
            f"{expected_dataset!r}"
        )
    if not isinstance(extra_info, Mapping):
        raise ValueError("extra_info must be a mapping")
    if extra_info.get("dataset") != dataset or extra_info.get("stage") != stage:
        raise ValueError("extra_info dataset/stage must match data_source")
    return stage, extra_info


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Mapping[str, Any] | None = None,
    **_: object,
) -> float:
    """VeRL custom reward entrypoint routed through the shared task reward."""
    stage, info = _route(data_source, extra_info)
    registry_path = os.environ.get(_REGISTRY_ENV, "").strip()
    if not registry_path:
        raise RuntimeError(f"{_REGISTRY_ENV} is required")
    registry = _load_registry(str(Path(registry_path).resolve()))
    if stage == "stage1":
        result = reward_stage1_choices(
            solution_str,
            ground_truth=ground_truth,
            registry=registry,
        )
    else:
        result = reward_stage2_choices(
            solution_str,
            ground_truth=ground_truth,
            candidates=info.get("candidates"),
            registry=registry,
        )
    return float(result.reward)


def _check_verl_version(allow_mismatch: bool) -> str:
    try:
        version = importlib_metadata.version("verl")
    except importlib_metadata.PackageNotFoundError as exc:
        raise RuntimeError("VeRL is not installed in the selected Python environment") from exc
    if version != _EXPECTED_VERL_VERSION and not allow_mismatch:
        raise RuntimeError(
            f"expected verl {_EXPECTED_VERL_VERSION}, found {version}; "
            "pass --allow-verl-version-mismatch only after compatibility review"
        )
    return version


def _environment(config: RlooExperimentConfig) -> dict[str, str]:
    environment = os.environ.copy()
    environment[_DATASET_ENV] = config.dataset
    environment[_REGISTRY_ENV] = str(config.registry_path.resolve())
    source_path = str(_REPO_ROOT / "src")
    existing_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = (
        source_path
        if not existing_pythonpath
        else source_path + os.pathsep + existing_pythonpath
    )
    environment.setdefault("TOKENIZERS_PARALLELISM", "false")
    environment.setdefault("HYDRA_FULL_ERROR", "1")
    return environment


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--task-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--rollout-n", type=int, default=2)
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--ppo-mini-batch-size", type=int, default=4)
    parser.add_argument("--ppo-micro-batch-size", type=int, default=1)
    parser.add_argument("--train-max-samples", type=int, default=8)
    parser.add_argument("--val-max-samples", type=int, default=2)
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--max-response-length", type=int, default=256)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--ppo-max-token-len-per-gpu", type=int, default=16384)
    parser.add_argument("--total-training-steps", type=int, default=3)
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--kl-coef", type=float, default=0.001)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--param-offload", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save-freq", type=int, default=1000)
    parser.add_argument("--test-freq", type=int, default=1000)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-verl-version-mismatch", action="store_true")
    return parser


def _config(args: argparse.Namespace) -> RlooExperimentConfig:
    return RlooExperimentConfig(
        dataset=args.dataset,
        model_path=args.model.expanduser().resolve(),
        data_dir=args.data_dir.expanduser().resolve(),
        registry_path=args.registry.expanduser().resolve(),
        corpus_path=args.corpus.expanduser().resolve(),
        task_config_path=args.task_config.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        python_bin=args.python_bin,
        rollout_n=args.rollout_n,
        train_batch_size=args.train_batch_size,
        ppo_mini_batch_size=args.ppo_mini_batch_size,
        ppo_micro_batch_size=args.ppo_micro_batch_size,
        train_max_samples=args.train_max_samples,
        val_max_samples=args.val_max_samples,
        max_prompt_length=args.max_prompt_length,
        max_response_length=args.max_response_length,
        max_model_len=args.max_model_len,
        ppo_max_token_len_per_gpu=args.ppo_max_token_len_per_gpu,
        total_training_steps=args.total_training_steps,
        actor_lr=args.actor_lr,
        kl_coef=args.kl_coef,
        gpu_memory_utilization=args.gpu_memory_utilization,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        enforce_eager=args.enforce_eager,
        param_offload=args.param_offload,
        save_freq=args.save_freq,
        test_freq=args.test_freq,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = _config(args)
    try:
        validation_command = build_validation_command(config)
        training_command = build_verl_command(config)
        if args.dry_run:
            payload = {
                "algorithm": "rloo",
                "server_validation_performed": False,
                "config": {key: str(value) for key, value in asdict(config).items()},
                "validation_command": validation_command,
                "training_command": training_command,
                "environment": {
                    _DATASET_ENV: config.dataset,
                    _REGISTRY_ENV: str(config.registry_path.resolve()),
                },
            }
            print(json.dumps(payload, indent=2, ensure_ascii=False))
            return 0

        config.validate_local_paths()
        _check_verl_version(args.allow_verl_version_mismatch)
        config.output_dir.mkdir(parents=True, exist_ok=True)
        os.chdir(_REPO_ROOT)
        validation_log = config.output_dir / "rl-contract-validation.log"
        with validation_log.open("w", encoding="utf-8", newline="\n") as handle:
            completed = subprocess.run(
                validation_command,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
                env=_environment(config),
            )
        if completed.returncode != 0:
            raise RuntimeError(
                f"RL contract validation failed; inspect {validation_log}"
            )
        os.execvpe(
            training_command[0],
            training_command,
            _environment(config),
        )
        raise AssertionError("os.execvpe returned unexpectedly")
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"rloo_experiment: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "RlooExperimentConfig",
    "build_validation_command",
    "build_verl_command",
    "compute_score",
    "main",
]
