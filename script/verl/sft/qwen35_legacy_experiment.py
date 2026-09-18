"""Fail-closed launcher for the shougang Qwen3.5 legacy two-stage SFT run."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from agent.hashing import sha256_file


FORMAL_TRAIN_ROWS = 8664
FORMAL_VAL_ROWS = 1082
REQUIRED_VERL_VERSION = "0.9.0"


@dataclass(frozen=True)
class ExperimentConfig:
    repo_root: Path
    python_bin: Path
    model_path: Path
    train_file: Path
    val_file: Path
    output_dir: Path
    mode: str = "formal"
    expected_train_rows: int | None = None
    expected_val_rows: int | None = None
    resume_from: Path | None = None
    seed: int = 42
    num_gpus: int = 1
    max_length: int = 4096
    lora_rank: int = 8
    lora_alpha: int = 16
    learning_rate: float = 1e-4
    effective_batch_size: int = 32
    micro_batch_size: int = 1

    def __post_init__(self) -> None:
        for field in (
            "repo_root", "python_bin", "model_path", "train_file", "val_file",
            "output_dir", "resume_from",
        ):
            value = getattr(self, field)
            if value is not None and not isinstance(value, Path):
                object.__setattr__(self, field, Path(value))
        if self.mode not in {"smoke", "formal"}:
            raise ValueError("mode must be smoke or formal")
        if self.num_gpus != 1:
            raise ValueError("the approved experiment requires exactly one GPU")
        if self.effective_batch_size % self.micro_batch_size:
            raise ValueError("effective batch size must be divisible by micro batch size")
        if self.resume_from is not None and self.mode == "smoke":
            raise ValueError("smoke mode does not resume checkpoints")

    @property
    def train_rows(self) -> int | None:
        if self.expected_train_rows is not None:
            return self.expected_train_rows
        return FORMAL_TRAIN_ROWS if self.mode == "formal" else None

    @property
    def val_rows(self) -> int | None:
        if self.expected_val_rows is not None:
            return self.expected_val_rows
        return FORMAL_VAL_ROWS if self.mode == "formal" else None

    def validate_output_policy(self) -> None:
        if self.resume_from is not None:
            if not self.resume_from.exists():
                raise FileNotFoundError(f"resume checkpoint not found: {self.resume_from}")
            return
        if self.output_dir.exists() and any(self.output_dir.iterdir()):
            raise FileExistsError(
                f"output directory is non-empty; use a fresh path: {self.output_dir}"
            )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "repo_root": str(self.repo_root),
            "python_bin": str(self.python_bin),
            "model_path": str(self.model_path),
            "train_file": str(self.train_file),
            "val_file": str(self.val_file),
            "output_dir": str(self.output_dir),
            "expected_train_rows": self.train_rows,
            "expected_val_rows": self.val_rows,
            "resume_from": str(self.resume_from) if self.resume_from else None,
            "seed": self.seed,
            "num_gpus": self.num_gpus,
            "max_length": self.max_length,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "target_modules": "all-linear",
            "learning_rate": self.learning_rate,
            "effective_batch_size": self.effective_batch_size,
            "micro_batch_size": self.micro_batch_size,
            "precision": "bfloat16",
            "epochs": 1,
        }


def validate_model_snapshot(model_path: str | Path) -> dict[str, Any]:
    root = Path(model_path)
    for required in ("config.json", "model.safetensors.index.json"):
        if not (root / required).is_file():
            raise FileNotFoundError(f"model file not found: {root / required}")
    index_path = root / "model.safetensors.index.json"
    with index_path.open(encoding="utf-8") as handle:
        index = json.load(handle)
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("model index requires a non-empty weight_map")
    shards = sorted(set(str(value) for value in weight_map.values()))
    missing = [name for name in shards if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError("missing model shard(s): " + ", ".join(missing))
    return {
        "index_sha256": sha256_file(index_path),
        "shard_count": len(shards),
        "total_shard_bytes": sum((root / name).stat().st_size for name in shards),
        "shards": {name: sha256_file(root / name) for name in shards},
    }


def validate_sft_parquet(
    path: str | Path, *, expected_rows: int | None = None
) -> dict[str, Any]:
    import pyarrow.parquet as pq

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"SFT parquet not found: {source}")
    table = pq.read_table(source, columns=["messages", "stage", "source_id"])
    rows = table.num_rows
    if expected_rows is not None and rows != expected_rows:
        raise ValueError(f"expected {expected_rows} rows in {source}, found {rows}")
    stages = table.column("stage").to_pylist()
    source_ids = table.column("source_id").to_pylist()
    messages = table.column("messages").to_pylist()
    stage_counts = Counter(str(stage) for stage in stages)
    if set(stage_counts) != {"stage1", "stage2"}:
        raise ValueError(f"unexpected stage values in {source}: {sorted(stage_counts)}")
    by_source: dict[str, list[str]] = defaultdict(list)
    for source_id, stage, row_messages in zip(source_ids, stages, messages):
        if not isinstance(source_id, str) or not source_id:
            raise ValueError(f"empty source_id in {source}")
        if not isinstance(row_messages, list) or len(row_messages) != 3:
            raise ValueError(f"every row must contain exactly three messages: {source}")
        by_source[source_id].append(str(stage))
    malformed = [
        source_id
        for source_id, source_stages in by_source.items()
        if sorted(source_stages) != ["stage1", "stage2"]
    ]
    if malformed:
        raise ValueError(f"source rows are not stage1/stage2 pairs: {malformed[:5]}")
    return {
        "file": str(source),
        "sha256": sha256_file(source),
        "rows": rows,
        "sources": len(by_source),
        "stage_counts": dict(sorted(stage_counts.items())),
    }


def build_command(config: ExperimentConfig) -> list[str]:
    run_script = config.repo_root / "script/verl/sft/run.sh"
    command = [
        "bash",
        str(run_script),
        f"data.train_files={config.train_file}",
        f"data.val_files={config.val_file}",
        "data.messages_key=messages",
        f"data.train_batch_size={2 if config.mode == 'smoke' else config.effective_batch_size}",
        f"data.micro_batch_size_per_gpu={config.micro_batch_size}",
        f"data.max_token_len_per_gpu={config.max_length}",
        f"data.max_length={config.max_length}",
        "data.truncation=error",
        "data.use_dynamic_bsz=false",
        f"data.num_workers={0 if config.mode == 'smoke' else 4}",
        f"model.path={config.model_path}",
        "model.use_remove_padding=false",
        f"model.enable_gradient_checkpointing={'false' if config.mode == 'smoke' else 'true'}",
        "+model.override_config.attn_implementation=sdpa",
        f"model.lora_rank={config.lora_rank}",
        f"model.lora_alpha={config.lora_alpha}",
        "model.target_modules=all-linear",
        "engine.strategy=fsdp",
        "engine.model_dtype=bfloat16",
        "engine.dtype=bfloat16",
        "engine.use_torch_compile=false",
        f"optim.lr={config.learning_rate}",
        "trainer.nnodes=1",
        "trainer.n_gpus_per_node=1",
        "trainer.project_name=dataclassify-sft",
        f"trainer.experiment_name=shougang-qwen35-9b-lora-{config.mode}",
        f"trainer.default_local_dir={config.output_dir / 'checkpoints'}",
        'trainer.logger=["console"]',
        "trainer.total_epochs=1",
        f"trainer.save_freq={1 if config.mode == 'smoke' else -1}",
        "trainer.max_ckpt_to_keep=2",
        "trainer.test_freq=-1",
        f"trainer.resume_mode={'resume_path' if config.resume_from else 'disable'}",
        f"trainer.resume_from_path={config.resume_from if config.resume_from else 'null'}",
    ]
    if config.mode == "smoke":
        command.append("trainer.total_training_steps=2")
    return command


def _probe_python(python_bin: Path) -> dict[str, Any]:
    code = (
        "import importlib.metadata as m,json,site,torch;"
        "import verl.trainer.sft_trainer;"
        "print(json.dumps({'verl':m.version('verl'),'torch':torch.__version__,"
        "'cuda':torch.version.cuda,'cuda_available':torch.cuda.is_available(),"
        "'site_packages':site.getsitepackages()[0]}))"
    )
    result = subprocess.run(
        [str(python_bin), "-c", code], capture_output=True, text=True, check=False
    )
    if result.returncode:
        raise RuntimeError(f"training Python probe failed: {result.stderr.strip()}")
    value = json.loads(result.stdout)
    if value["verl"] != REQUIRED_VERL_VERSION:
        raise ValueError(
            f"VeRL must be {REQUIRED_VERL_VERSION}, found {value['verl']}"
        )
    if not value["cuda_available"]:
        raise RuntimeError("CUDA is unavailable in the training environment")
    return value


def validate_runtime(config: ExperimentConfig) -> dict[str, Any]:
    config.validate_output_policy()
    run_script = config.repo_root / "script/verl/sft/run.sh"
    if not run_script.is_file():
        raise FileNotFoundError(f"SFT launcher not found: {run_script}")
    model = validate_model_snapshot(config.model_path)
    train = validate_sft_parquet(config.train_file, expected_rows=config.train_rows)
    val = validate_sft_parquet(config.val_file, expected_rows=config.val_rows)
    environment = _probe_python(config.python_bin)
    verifier = config.repo_root / "script/verl/common/patches/verify_verl_patches.sh"
    env = os.environ.copy()
    env["PATH"] = str(config.python_bin.parent) + os.pathsep + env.get("PATH", "")
    patch_result = subprocess.run(
        ["bash", str(verifier), environment["site_packages"]],
        cwd=config.repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if patch_result.returncode:
        raise RuntimeError(
            "required VeRL patches did not verify:\n" + patch_result.stderr.strip()
        )
    return {
        "model": model,
        "train": train,
        "validation": val,
        "environment": environment,
        "patch_verification": patch_result.stdout,
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--python-bin", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--val-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    parser.add_argument("--expected-train-rows", type=int)
    parser.add_argument("--expected-val-rows", type=int)
    parser.add_argument("--resume-from")
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = ExperimentConfig(
        repo_root=Path(args.repo_root),
        python_bin=Path(args.python_bin),
        model_path=Path(args.model_path),
        train_file=Path(args.train_file),
        val_file=Path(args.val_file),
        output_dir=Path(args.output_dir),
        mode=args.mode,
        expected_train_rows=args.expected_train_rows,
        expected_val_rows=args.expected_val_rows,
        resume_from=Path(args.resume_from) if args.resume_from else None,
        micro_batch_size=args.micro_batch_size,
    )
    audit = validate_runtime(config)
    payload = {**config.to_mapping(), "audit": audit, "command": build_command(config)}
    config_file = config.output_dir / "resolved-config.json"
    _write_json(config_file, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.dry_run:
        return 0
    env = os.environ.copy()
    env["PYTHON_BIN"] = str(config.python_bin)
    env["NUM_GPUS"] = str(config.num_gpus)
    completed = subprocess.run(
        build_command(config), cwd=config.repo_root, env=env, check=False
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
