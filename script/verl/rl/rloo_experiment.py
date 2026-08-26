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
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Mapping, Sequence

from agent.task import GradingConfig, LeafRegistry
from agent.task.grading_manifest import DatasetGradingManifest
from agent.training.rl.cascade import (
    CASCADE_K,
    CASCADE_N,
    CascadeConfig,
    cascade_reward,
    decode_stage1_candidates,
    decode_stage2_output,
)
from agent.training.rl.reward import reward_stage1_choices, reward_stage2_choices

_EXPECTED_VERL_VERSION = "0.8.0"
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DATASET_ENV = "DATACLASSIFY_RLOO_DATASET"
_REGISTRY_ENV = "DATACLASSIFY_RLOO_REGISTRY"
_CORPUS_ENV = "DATACLASSIFY_RLOO_CORPUS"
_GRADING_MANIFEST_ENV = "DATACLASSIFY_RLOO_GRADING_MANIFEST"
_AGENT_LOOP_CONFIG = _REPO_ROOT / "cfg" / "verl" / "rl" / "cascade_agent_loop.yaml"
_RELEASE_DATASETS = frozenset({"finance", "shougang"})
_GPU_TARGET = "RTX PRO 6000 96GB"
_EXACT_VERSION = re.compile(r"^v?\d+\.\d+\.\d+(?:[.+-][0-9A-Za-z.-]+)?$")


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
    grading_manifest_path: Path | None = None
    python_bin: str = sys.executable
    rollout_n: int = 4
    train_batch_size: int = 4
    ppo_mini_batch_size: int = 4
    ppo_micro_batch_size: int = 1
    train_max_samples: int = 8
    val_max_samples: int = 2
    max_prompt_length: int = 7168
    # Multi-turn response space contains Stage1 assistant tokens, the masked
    # dynamic Stage2 user bridge, and Stage2 assistant tokens.
    max_response_length: int = 2048
    max_model_len: int = 12288
    ppo_max_token_len_per_gpu: int = 16384
    total_training_steps: int = 3
    actor_lr: float = 1e-4
    # Actor KL loss is the only KL term in the fixed policy.  KL in reward
    # remains disabled by build_verl_command.
    kl_coef: float = 0.001
    gpu_memory_utilization: float = 0.5
    lora_rank: int = 8
    lora_alpha: int = 16
    enforce_eager: bool = False
    param_offload: bool = False
    save_freq: int = 1
    test_freq: int = 1
    max_ckpt_keep: int = 1
    experiment_name: str | None = None
    resume_mode: str = "auto"
    reference_provenance_path: Path | None = None
    reference_checkpoint_sha256: str | None = None
    vllm_version: str | None = None
    prompt_budget_chars: int = 200_000
    gpu_target: str = _GPU_TARGET

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
            "save_freq": self.save_freq,
            "max_ckpt_keep": self.max_ckpt_keep,
            "test_freq": self.test_freq,
            "prompt_budget_chars": self.prompt_budget_chars,
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
        if self.save_freq > self.total_training_steps:
            raise ValueError("save_freq must be reachable within total_training_steps")
        if self.test_freq > self.total_training_steps:
            raise ValueError("test_freq must be reachable within total_training_steps")
        if self.resume_mode not in {"auto", "resume", "disable"}:
            raise ValueError("resume_mode must be auto, resume, or disable")
        if self.gpu_target != _GPU_TARGET:
            raise ValueError(f"gpu target is fixed to {_GPU_TARGET}")
        if self.vllm_version is not None and not _EXACT_VERSION.fullmatch(self.vllm_version.strip()):
            raise ValueError("vllm_version must be an exact frozen version")
        if self.reference_checkpoint_sha256 is not None and not re.fullmatch(
            r"[0-9a-fA-F]{64}", self.reference_checkpoint_sha256.strip()
        ):
            raise ValueError("reference_checkpoint_sha256 must be a 64-character SHA-256")

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "run_manifest.json"

    @property
    def validation_report_path(self) -> Path:
        return self.output_dir / "cascade-validation-report.json"

    @property
    def release_datasets(self) -> tuple[str, ...]:
        values = tuple(
            item.strip().lower()
            for item in re.split(r"[+,]", self.dataset)
            if item.strip()
        )
        return values

    def validate_release_policy(self) -> None:
        release_datasets = self.release_datasets
        if (
            len(release_datasets) != len(set(release_datasets))
            or set(release_datasets) != _RELEASE_DATASETS
        ):
            raise ValueError(
                "formal RLOO is restricted to one finance+shougang joint release"
            )
        if self.rollout_n != CASCADE_N:
            raise ValueError(f"formal cascade rollout N is fixed at {CASCADE_N}")
        if self.kl_coef != 0.001:
            raise ValueError("actor KL loss coefficient is fixed at 0.001")
        if self.resume_mode == "disable":
            raise ValueError("formal RLOO requires a reachable resume mode")
        if self.reference_provenance_path is None:
            raise ValueError("reference provenance JSON is required for formal RLOO")
        if self.grading_manifest_path is None:
            raise ValueError("formal cascade RLOO requires a grading manifest")
        if self.vllm_version is None:
            raise ValueError("vLLM exact frozen version is required; no unverified pin is assumed")

    def validate_local_paths(self) -> None:
        required_files = (
            self.train_file,
            self.val_file,
            self.test_file,
            self.registry_path,
            self.corpus_path,
            self.task_config_path,
            *(
                ()
                if self.grading_manifest_path is None
                else (self.grading_manifest_path,)
            ),
        )
        for path in required_files:
            if not path.is_file():
                raise FileNotFoundError(f"required local file not found: {path}")
        if not (self.model_path / "config.json").is_file():
            raise FileNotFoundError(f"model config not found: {self.model_path / 'config.json'}")
        if self.reference_provenance_path is not None and not self.reference_provenance_path.is_file():
            raise FileNotFoundError(
                f"reference provenance not found: {self.reference_provenance_path}"
            )
        if not str(self.output_dir).strip():
            raise ValueError("output_dir must be non-empty")


def _bool(value: bool) -> str:
    return "True" if value else "False"


def sqrt_sampling_probabilities(counts: Mapping[str, int] | Sequence[int]) -> dict[str, float] | tuple[float, ...]:
    """Return dataset sampling probabilities proportional to ``sqrt(n)``."""

    if isinstance(counts, Mapping):
        values = {str(name): int(count) for name, count in counts.items()}
        if not values or any(count < 0 for count in values.values()):
            raise ValueError("dataset counts must be non-negative and non-empty")
        total = sum(value ** 0.5 for value in values.values())
        if total <= 0:
            raise ValueError("at least one dataset count must be positive")
        return {name: (count ** 0.5) / total for name, count in values.items()}
    values = tuple(int(count) for count in counts)
    if not values or any(count < 0 for count in values):
        raise ValueError("dataset counts must be non-negative and non-empty")
    total = sum(value ** 0.5 for value in values)
    if total <= 0:
        raise ValueError("at least one dataset count must be positive")
    return tuple((count ** 0.5) / total for count in values)


def build_validation_command(config: RlooExperimentConfig) -> list[str]:
    """Build the formal Stage1-only cascade mixture validation command."""
    config.validate_options()
    if config.grading_manifest_path is None:
        raise ValueError("cascade validation requires a grading manifest")
    return [
        config.python_bin,
        "-m",
        "script.verl.rl.validate_cascade",
        "--dataset-dir",
        str(config.data_dir),
        "--registry",
        str(config.registry_path),
        "--corpus",
        str(config.corpus_path),
        "--task-config",
        str(config.task_config_path),
        "--grading-manifest",
        str(config.grading_manifest_path),
        "--report",
        str(config.validation_report_path),
    ]


def build_verl_command(config: RlooExperimentConfig) -> list[str]:
    """Build the VeRL command; all RLOO math remains inside VeRL."""
    config.validate_options()
    reward_adapter = "pkg://" + Path(__file__).resolve().relative_to(_REPO_ROOT).with_suffix("").as_posix()
    overrides = [
        "algorithm.adv_estimator=rloo",
        # Fixed policy: KL is an actor loss only; never add KL to reward.
        "algorithm.use_kl_in_reward=False",
        "algorithm.kl_penalty=kl",
        "algorithm.kl_ctrl.kl_coef=0.0",
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
        "+actor_rollout_ref.model.override_config.attn_implementation=flash_attention_2",
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
        "actor_rollout_ref.actor.use_kl_loss=True",
        f"actor_rollout_ref.actor.kl_loss_coef={config.kl_coef}",
        "actor_rollout_ref.actor.entropy_coeff=0",
        f"actor_rollout_ref.actor.optim.lr={config.actor_lr}",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.mode=async",
        "actor_rollout_ref.rollout.multi_turn.enable=True",
        "actor_rollout_ref.rollout.agent.default_agent_loop=dataclassify_cascade",
        f"actor_rollout_ref.rollout.agent.agent_loop_config_path={_AGENT_LOOP_CONFIG}",
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
        f"trainer.experiment_name={config.experiment_name or 'rloo-smoke-' + config.dataset}",
        f"trainer.default_local_dir={config.output_dir / 'checkpoints'}",
        "trainer.total_epochs=4",
        f"trainer.total_training_steps={config.total_training_steps}",
        f"trainer.save_freq={config.save_freq}",
        f"trainer.max_actor_ckpt_to_keep={config.max_ckpt_keep}",
        f"trainer.test_freq={config.test_freq}",
        "trainer.val_before_train=False",
        f"trainer.resume_mode={config.resume_mode}",
    ]
    return [config.python_bin, "-m", "verl.trainer.main_ppo", *overrides]


def _check_reference_provenance(config: RlooExperimentConfig) -> dict[str, Any]:
    """Verify the released merged HF reference through the shared SFT seam."""

    if config.reference_provenance_path is None:
        raise RuntimeError("reference provenance JSON is required")
    path = config.reference_provenance_path
    if not path.is_file():
        raise FileNotFoundError(f"reference provenance not found: {path}")
    try:
        # Keep provenance policy in one place: this verifier recomputes the
        # merged-HF tree hash and validates export/config/environment lineage.
        from script.verl.sft.record_checkpoint import verify_reference_provenance

        value = verify_reference_provenance(path, config.model_path)
    except (ImportError, OSError, ValueError, TypeError, RuntimeError) as exc:
        raise RuntimeError(f"reference provenance verification failed: {exc}") from exc
    if not isinstance(value, Mapping):
        raise RuntimeError("reference provenance verifier returned a non-object")
    digest = value.get("checkpoint_sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
        raise RuntimeError("reference provenance must carry a 64-character checkpoint_sha256")
    if config.reference_checkpoint_sha256 and digest.lower() != config.reference_checkpoint_sha256.lower():
        raise RuntimeError("reference checkpoint SHA-256 does not match provenance")
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "checkpoint_sha256": digest.lower(),
        "algorithm": value.get("algorithm"),
        "artifact_kind": value.get("artifact_kind"),
        "verified": True,
    }


def _check_selected_python(config: RlooExperimentConfig) -> dict[str, Any]:
    """Verify dependencies in the *selected* interpreter, not this process."""

    code = (
        "import importlib, importlib.metadata as m, json; "
        "names=['verl','vllm','ray','torch']; "
        "out={}; "
        "[(__import__(name), out.__setitem__(name, m.version(name))) for name in names]; "
        "print(json.dumps(out))"
    )
    try:
        completed = subprocess.run(
            [config.python_bin, "-c", code],
            capture_output=True,
            text=True,
            check=False,
            env=_environment(config),
        )
    except OSError as exc:
        raise RuntimeError(f"cannot execute selected Python {config.python_bin!r}") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "dependency probe failed").strip()
        raise RuntimeError(
            "selected Python must provide verl, vllm, ray, and torch: " + detail
        )
    try:
        versions = json.loads(completed.stdout.strip())
    except (json.JSONDecodeError, TypeError) as exc:
        raise RuntimeError("selected Python dependency probe returned invalid JSON") from exc
    if not isinstance(versions, Mapping):
        raise RuntimeError("selected Python dependency probe returned a non-object")
    if versions.get("verl") != _EXPECTED_VERL_VERSION:
        raise RuntimeError(
            f"selected Python must have verl {_EXPECTED_VERL_VERSION}, found {versions.get('verl')!r}"
        )
    if config.vllm_version is None:
        raise RuntimeError("vLLM exact frozen version is required; preflight will not infer a pin")
    if versions.get("vllm") != config.vllm_version:
        raise RuntimeError(
            f"selected Python vllm {versions.get('vllm')!r} does not match frozen {config.vllm_version!r}"
        )
    for name in ("ray", "torch"):
        if not versions.get(name):
            raise RuntimeError(f"selected Python dependency {name} has no version")
    return {"python": config.python_bin, "versions": dict(versions)}


def _check_prompt_budget(config: RlooExperimentConfig) -> dict[str, Any]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("prompt-budget preflight requires pyarrow") from exc
    details: dict[str, Any] = {"budget_chars": config.prompt_budget_chars, "files": {}}
    for split, path in (("train", config.train_file), ("val", config.val_file), ("test", config.test_file)):
        if not path.is_file():
            raise FileNotFoundError(f"required local file not found: {path}")
        try:
            rows = pq.read_table(path).to_pylist()
        except Exception as exc:
            raise RuntimeError(f"cannot read {split} parquet for prompt preflight") from exc
        if not rows:
            raise RuntimeError(f"{split} parquet must contain at least one row")
        max_chars = 0
        for index, row in enumerate(rows):
            prompt = row.get("prompt") if isinstance(row, Mapping) else None
            if not isinstance(prompt, list) or not prompt:
                raise RuntimeError(f"{split} row {index} has an empty prompt")
            chars = sum(
                len(message.get("content", ""))
                for message in prompt
                if isinstance(message, Mapping)
            )
            if chars <= 0:
                raise RuntimeError(f"{split} row {index} has an empty prompt")
            max_chars = max(max_chars, chars)
            if chars > config.prompt_budget_chars:
                raise RuntimeError(
                    f"{split} row {index} prompt has {chars} chars, over budget {config.prompt_budget_chars}"
                )
        details["files"][split] = {"rows": len(rows), "max_chars": max_chars}
    return details


def validate_preflight(
    config: RlooExperimentConfig,
    *,
    check_runtime: bool = True,
) -> dict[str, Any]:
    """Run deterministic local gates before any server/GPU process starts.

    This function performs no CUDA probe and makes no GPU claim; server-side
    validation remains a separate gate.  CPU tests pass ``check_runtime=False``
    or inject a subprocess probe.
    """

    config.validate_options()
    config.validate_release_policy()
    config.validate_local_paths()
    assets = _check_formal_assets(config)
    reference = _check_reference_provenance(config)
    prompt_budget = _check_prompt_budget(config)
    runtime = _check_selected_python(config) if check_runtime else {
        "python": config.python_bin,
        "versions": None,
        "runtime_check_skipped": True,
    }
    output_parent = config.output_dir.parent
    output_parent.mkdir(parents=True, exist_ok=True)
    if config.output_dir.exists() and not config.output_dir.is_dir():
        raise RuntimeError(f"output path is not a directory: {config.output_dir}")
    return {
        "passed": True,
        "dataset": config.dataset,
        "release": "finance+shougang",
        "sampling": "p proportional to sqrt(n)",
        "cascade": {"k": CASCADE_K, "n": CASCADE_N, "direct": True},
        "actor": {"lora": True, "dtype": "bfloat16", "kl_loss_coef": config.kl_coef},
        "kl_in_reward": False,
        "gpu_target": config.gpu_target,
        "gpu_validation_performed": False,
        "assets": assets,
        "reference": reference,
        "prompt_budget": prompt_budget,
        "runtime": runtime,
        "reachable": {
            "total_training_steps": config.total_training_steps,
            "save_freq": config.save_freq,
            "max_ckpt_keep": config.max_ckpt_keep,
            "test_freq": config.test_freq,
            "resume_mode": config.resume_mode,
            "manifest": str(config.manifest_path),
        },
    }


def write_run_manifest(
    config: RlooExperimentConfig,
    preflight: Mapping[str, Any],
    validation_command: Sequence[str],
    training_command: Sequence[str],
    *,
    validation_report_path: Path | None = None,
) -> Path:
    """Persist the preflight/provenance gate before launching VeRL.

    The first write records the commands before validation starts.  The
    launcher writes the same manifest again after the validator succeeds with
    ``validation_report`` path/hash lineage bound to this run.
    """

    config.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "dataclassify-rloo-run-manifest-v1",
        "config": {key: str(value) for key, value in asdict(config).items()},
        "preflight": dict(preflight),
        "validation_command": list(validation_command),
        "training_command": list(training_command),
    }
    if validation_report_path is not None:
        report_path = Path(validation_report_path).expanduser().resolve()
        expected_path = config.validation_report_path.expanduser().resolve()
        if report_path != expected_path:
            raise ValueError(
                "validation report must be the configured cascade report path"
            )
        if not report_path.is_file():
            raise FileNotFoundError(f"validation report not found: {report_path}")
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("validation report is not valid JSON") from exc
        if not isinstance(report, Mapping) or report.get("valid") is not True:
            raise RuntimeError("validation report did not pass the cascade contract")
        payload["validation_report"] = {
            "path": str(report_path),
            "sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        }
    config.manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not config.manifest_path.read_text(encoding="utf-8").strip():
        raise RuntimeError("run manifest was empty after write")
    return config.manifest_path


@lru_cache(maxsize=8)
def _load_registry(path: str) -> LeafRegistry:
    return LeafRegistry.from_path(path)


@lru_cache(maxsize=8)
def _load_grading_manifest(path: str) -> DatasetGradingManifest:
    return DatasetGradingManifest.from_path(path)


def _check_formal_assets(config: RlooExperimentConfig) -> dict[str, Any]:
    """Validate immutable prompt/rubric assets before any server starts."""

    registry = LeafRegistry.from_path(config.registry_path)
    missing_descriptions = sorted(
        category.category_id
        for category in registry.categories
        if not isinstance(category.description, str)
        or not category.description.strip()
    )
    if missing_descriptions:
        raise ValueError(
            "formal cascade registry entries require non-empty descriptions: "
            + ", ".join(missing_descriptions)
        )
    if config.grading_manifest_path is None:
        raise ValueError("formal cascade grading manifest is required")
    grading_manifest = DatasetGradingManifest.from_path(
        config.grading_manifest_path
    )
    invalid_gt_field = sorted(
        dataset
        for dataset in ("finance", "shougang")
        if grading_manifest.config_for(dataset).gt_field != "data_level"
    )
    if invalid_gt_field:
        raise ValueError(
            "formal cascade grading gt_field must be data_level for: "
            + ", ".join(invalid_gt_field)
        )
    return {
        "registry": str(config.registry_path),
        "registry_descriptions": "non-empty",
        "grading_manifest": str(config.grading_manifest_path),
        "grading_gt_field": "data_level",
    }


def _configured_datasets() -> frozenset[str]:
    raw = os.environ.get(_DATASET_ENV, "")
    values = frozenset(
        item.strip().lower()
        for item in re.split(r"[+,]", raw)
        if item.strip()
    )
    if not values:
        raise RuntimeError(f"{_DATASET_ENV} is required")
    return values


def _route(data_source: str, extra_info: Mapping[str, Any] | None) -> tuple[str, Mapping[str, Any]]:
    if not isinstance(data_source, str):
        raise ValueError("data_source must be a string")
    try:
        dataset, stage = data_source.rsplit("/", 1)
    except ValueError:
        raise ValueError("data_source must be '<dataset>/stage1|stage2'") from None
    if stage not in {"stage1", "stage2"} or not dataset:
        raise ValueError("data_source must be '<dataset>/stage1|stage2'")
    expected_datasets = _configured_datasets()
    if dataset not in expected_datasets:
        configured = ",".join(sorted(expected_datasets))
        raise ValueError(
            f"data_source dataset {dataset!r} does not match configured dataset(s) "
            f"{configured!r}"
        )
    if not isinstance(extra_info, Mapping):
        raise ValueError("extra_info must be a mapping")
    if extra_info.get("dataset") != dataset or extra_info.get("stage") != stage:
        raise ValueError("extra_info dataset/stage must match data_source")
    return stage, extra_info


def _json_objects(solution: str) -> tuple[Mapping[str, Any], ...]:
    """Extract object-shaped JSON values from a multi-turn response.

    The direct AgentLoop response contains Stage 1 JSON, a natural-language
    bridge, and Stage 2 JSON.  VeRL's static reward callback receives the
    decoded concatenation when a pre-computed ``reward_score`` is unavailable;
    scanning complete objects gives that fallback a deterministic, bounded
    parser without treating bridge prose as model output.
    """

    if not isinstance(solution, str):
        return ()
    decoder = json.JSONDecoder()
    values: list[Mapping[str, Any]] = []
    for index, character in enumerate(solution):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(solution, index)
        except json.JSONDecodeError:
            continue
        if isinstance(value, Mapping):
            values.append(value)
        if len(values) >= 32:
            break
    return tuple(values)


def _static_cascade_fallback(
    solution: str,
    *,
    ground_truth: str,
    ground_truth_level: str | None,
    registry: LeafRegistry,
    grading: GradingConfig,
) -> float | None:
    """Score a complete direct response for reward-manager fallback."""

    if ground_truth not in registry.ids:
        return 0.0
    stage1_object = next(
        (
            value
            for value in _json_objects(solution)
            if set(value) == {"candidates"}
        ),
        None,
    )
    if stage1_object is None:
        return None
    stage1_solution = json.dumps(stage1_object, separators=(",", ":"))
    try:
        candidates, stage1_errors = decode_stage1_candidates(
            stage1_solution, registry=registry
        )
    except (TypeError, ValueError):
        return 0.0
    if stage1_errors:
        return 0.0
    stage2_object = next(
        (
            value
            for value in _json_objects(solution)
            if set(value) == {"answer", "level"}
            and isinstance(value.get("answer"), str)
            and isinstance(value.get("level"), str)
            and value.get("level") in grading.levels
        ),
        None,
    )
    if stage2_object is None:
        # A standalone Stage 1 response remains scoreable by the static task
        # reward; the direct-cascade reward is used only once both objects are
        # present.
        return float(
            reward_stage1_choices(
                stage1_solution,
                ground_truth=ground_truth,
                registry=registry,
            ).reward
        )
    stage2_solution = json.dumps(stage2_object, separators=(",", ":"))
    leaf, level, stage2_errors = decode_stage2_output(
        stage2_solution, candidates=candidates, grading=grading
    )
    if stage2_errors:
        return 0.0
    return cascade_reward(
        stage1_hit=ground_truth in candidates,
        leaf_correct=leaf == ground_truth,
        level_correct=(ground_truth_level is not None and level == ground_truth_level),
        valid=True,
        config=CascadeConfig(),
    )


def compute_cascade_score(
    stage1_solution: str,
    stage2_solution: str,
    *,
    ground_truth: str,
    ground_truth_level: str | None,
    registry: LeafRegistry,
    grading: "GradingConfig | None" = None,
) -> float:
    """Score one complete direct-cascade trajectory.

    This helper is backend agnostic and is useful to VeRL reward adapters as
    well as CPU fake-rollout tests.  Stage 2 is decoded only against the
    actual Stage 1 candidates; malformed either segment yields zero.
    """

    if grading is None or ground_truth not in registry.ids:
        # The formal cascade always has a dataset-specific grading rubric;
        # silently dropping to classification-only scoring would make a
        # malformed release look successful.  Return the safe reward instead
        # of raising from a training reward callback.
        return 0.0
    try:
        candidates, stage1_errors = decode_stage1_candidates(
            stage1_solution, registry=registry
        )
    except (TypeError, ValueError):
        return 0.0
    if stage1_errors:
        return 0.0
    try:
        leaf, level, stage2_errors = decode_stage2_output(
            stage2_solution, candidates=candidates, grading=grading
        )
    except (TypeError, ValueError):
        return 0.0
    if stage2_errors:
        return 0.0
    return cascade_reward(
        stage1_hit=ground_truth in candidates,
        leaf_correct=leaf == ground_truth,
        level_correct=(ground_truth_level is not None and level == ground_truth_level),
        valid=True,
        config=CascadeConfig(),
    )


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
    manifest_path = os.environ.get(_GRADING_MANIFEST_ENV, "").strip()
    manifest_grading: GradingConfig | None = None
    if manifest_path:
        level = info.get("ground_truth_level")
        if not isinstance(level, str) or not level.strip():
            raise ValueError(
                "formal reward requires extra_info.ground_truth_level"
            )
        info_level = level.strip()
        manifest_grading = _load_grading_manifest(
            str(Path(manifest_path).resolve())
        ).config_for(info["dataset"])
        if info_level not in manifest_grading.levels:
            raise ValueError(
                f"ground_truth_level {info_level!r} is outside the approved "
                f"{info['dataset']} rubric"
            )
    else:
        info_level = None
    if stage == "stage1":
        result = reward_stage1_choices(
            solution_str,
            ground_truth=ground_truth,
            registry=registry,
        )
        if manifest_grading is not None and not result.format_valid:
            fallback = _static_cascade_fallback(
                solution_str,
                ground_truth=ground_truth,
                ground_truth_level=info_level,
                registry=registry,
                grading=manifest_grading,
            )
            if fallback is not None:
                return float(fallback)
    else:
        grading: GradingConfig | None = manifest_grading
        expected_level: str | None = info_level
        if grading is None and info["dataset"] in _RELEASE_DATASETS:
            raise RuntimeError(
                f"{_GRADING_MANIFEST_ENV} is required for formal Stage 2 reward"
            )
        result = reward_stage2_choices(
            solution_str,
            ground_truth=ground_truth,
            candidates=info.get("candidates"),
            registry=registry,
            grading=grading,
            expected_level=expected_level,
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
    environment[_CORPUS_ENV] = str(config.corpus_path.resolve())
    if config.grading_manifest_path is not None:
        environment[_GRADING_MANIFEST_ENV] = str(
            config.grading_manifest_path.resolve()
        )
    source_path = str(_REPO_ROOT / "src")
    existing_pythonpath = environment.get("PYTHONPATH", "")
    # script.* lives at the repo root (not under src); Ray workers do not
    # inherit the launcher cwd, so repo root must be on PYTHONPATH too.
    root_path = str(_REPO_ROOT)
    environment["PYTHONPATH"] = os.pathsep.join(
        part
        for part in (root_path, source_path, existing_pythonpath)
        if part
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
    parser.add_argument("--grading-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--rollout-n", type=int, default=4)
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--ppo-mini-batch-size", type=int, default=4)
    parser.add_argument("--ppo-micro-batch-size", type=int, default=1)
    parser.add_argument("--train-max-samples", type=int, default=8)
    parser.add_argument("--val-max-samples", type=int, default=2)
    parser.add_argument("--max-prompt-length", type=int, default=7168)
    parser.add_argument("--max-response-length", type=int, default=2048)
    parser.add_argument("--max-model-len", type=int, default=12288)
    parser.add_argument("--ppo-max-token-len-per-gpu", type=int, default=16384)
    parser.add_argument("--total-training-steps", type=int, default=3)
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--kl-coef", type=float, default=0.001)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--param-offload", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save-freq", type=int, default=1)
    parser.add_argument("--max-ckpt-keep", type=int, default=1)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--test-freq", type=int, default=1)
    parser.add_argument("--resume-mode", choices=("auto", "resume", "disable"), default="auto")
    parser.add_argument("--reference-provenance", type=Path)
    parser.add_argument("--reference-checkpoint-sha256")
    parser.add_argument(
        "--vllm-version",
        help="Exact frozen vLLM version from the VeRL 0.8 compatibility lock",
    )
    parser.add_argument("--prompt-budget-chars", type=int, default=200_000)
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
        grading_manifest_path=(
            args.grading_manifest.expanduser().resolve()
            if args.grading_manifest is not None
            else None
        ),
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
        max_ckpt_keep=args.max_ckpt_keep,
        experiment_name=args.experiment_name,
        resume_mode=args.resume_mode,
        reference_provenance_path=(
            args.reference_provenance.expanduser().resolve()
            if args.reference_provenance is not None
            else None
        ),
        reference_checkpoint_sha256=args.reference_checkpoint_sha256,
        vllm_version=args.vllm_version,
        prompt_budget_chars=args.prompt_budget_chars,
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

        preflight = validate_preflight(config, check_runtime=True)
        _check_verl_version(args.allow_verl_version_mismatch)
        config.output_dir.mkdir(parents=True, exist_ok=True)
        write_run_manifest(config, preflight, validation_command, training_command)
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
        # Bind the validator's structured artifact only after a successful
        # validation process.  This prevents a preflight-only manifest from
        # being mistaken for a formally validated run.
        write_run_manifest(
            config,
            preflight,
            validation_command,
            training_command,
            validation_report_path=config.validation_report_path,
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
    "sqrt_sampling_probabilities",
    "validate_preflight",
    "write_run_manifest",
    "compute_score",
    "compute_cascade_score",
    "main",
]
