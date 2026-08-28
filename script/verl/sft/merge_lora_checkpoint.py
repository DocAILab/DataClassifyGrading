"""Merge a verified single-rank VeRL LoRA checkpoint into a standalone HF model.

The output is published atomically only after a fresh Transformers load of
both model and tokenizer succeeds.  The current implementation deliberately
supports only the server-validated ``world_size=1, rank=0`` checkpoint layout;
other layouts fail closed instead of guessing how shards should be merged.
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

LORA_A = re.compile(r"(.*)\.lora_A\.default\.weight$")
_MODEL_FILE = re.compile(r"model_world_size_(\d+)_rank_(\d+)\.pt$")


def discover_single_rank_checkpoint(checkpoint: str | Path) -> tuple[Path, Mapping[str, Any]]:
    """Return the sole rank-0 model file and required LoRA metadata."""
    root = Path(checkpoint)
    if not root.is_dir():
        raise NotADirectoryError(f"checkpoint directory not found: {root}")
    model_files = sorted(
        path for path in root.iterdir() if path.is_file() and _MODEL_FILE.fullmatch(path.name)
    )
    if len(model_files) != 1:
        raise ValueError(
            "checkpoint must contain exactly one model_world_size_*_rank_*.pt file"
        )
    match = _MODEL_FILE.fullmatch(model_files[0].name)
    assert match is not None
    world_size, rank = int(match.group(1)), int(match.group(2))
    if world_size != 1 or rank != 0:
        raise ValueError(
            f"only the verified world_size=1 rank=0 layout is supported, got "
            f"world_size={world_size} rank={rank}"
        )
    meta_file = root / "lora_train_meta.json"
    if not meta_file.is_file():
        raise FileNotFoundError(f"lora_train_meta.json not found: {meta_file}")
    try:
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid lora_train_meta.json: {exc}") from exc
    if not isinstance(meta, Mapping):
        raise ValueError("lora_train_meta.json must be a JSON object")
    return model_files[0], meta


def lora_spec(keys: Sequence[str], meta: Mapping[str, Any]) -> tuple[list[str], int, int]:
    """Validate checkpoint metadata and return target modules/rank/alpha."""
    targets = sorted(
        {
            match.group(1).rsplit(".", 1)[-1]
            for key in keys
            for match in [LORA_A.match(key)]
            if match
        }
    )
    if not targets:
        raise ValueError("no LoRA keys found in checkpoint")
    rank = meta.get("r")
    alpha = meta.get("lora_alpha")
    if (
        isinstance(rank, bool)
        or not isinstance(rank, int)
        or rank <= 0
        or isinstance(alpha, bool)
        or not isinstance(alpha, int)
        or alpha <= 0
    ):
        raise ValueError("LoRA r and lora_alpha must be positive integers")
    return targets, rank, alpha


def require_new_output_dir(output: str | Path) -> Path:
    """Return *output* only when no file or directory already occupies it."""
    path = Path(output)
    if path.exists():
        raise FileExistsError(f"merge output must not already exist: {path}")
    return path


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True,
                        help="VeRL single-rank global_step checkpoint directory")
    parser.add_argument("--base-model", required=True,
                        help="Exact HF base model directory used by SFT")
    parser.add_argument("--output", required=True,
                        help="New merged HF output directory")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        output = require_new_output_dir(args.output)
        model_file, meta = discover_single_rank_checkpoint(args.checkpoint)
    except (FileNotFoundError, NotADirectoryError, ValueError, FileExistsError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    import torch
    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

    state_dict = torch.load(model_file, map_location="cpu", weights_only=False)
    if not isinstance(state_dict, Mapping):
        print("error: checkpoint state must be a mapping", file=sys.stderr)
        return 2
    try:
        targets, rank, alpha = lora_spec(list(state_dict), meta)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    base_path = Path(args.base_model)
    if not base_path.is_dir():
        print(f"error: base model directory not found: {base_path}", file=sys.stderr)
        return 2
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.merge-", dir=output.parent))
    try:
        print(f"[merge] targets={targets} r={rank} alpha={alpha} keys={len(state_dict)}")
        base = AutoModelForImageTextToText.from_pretrained(
            base_path, torch_dtype=torch.bfloat16, local_files_only=True
        )
        config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            target_modules=targets,
            task_type="CAUSAL_LM",
        )
        peft = PeftModel(base, config)
        _, unexpected = peft.load_state_dict(state_dict, strict=False)
        unexpected = list(unexpected)
        used = len(state_dict) - len(unexpected)
        ratio = used / len(state_dict)
        if unexpected:
            print(
                f"warning: {len(unexpected)} unexpected checkpoint keys: "
                f"{unexpected[:5]}"
            )
        print(f"[merge] consumed {used}/{len(state_dict)} checkpoint keys ({ratio:.1%})")
        if ratio < 0.99:
            raise ValueError("checkpoint keys consumed < 99%; aborting")

        merged = peft.merge_and_unload()
        merged.save_pretrained(staging)
        tokenizer = AutoTokenizer.from_pretrained(base_path, local_files_only=True)
        tokenizer.save_pretrained(staging)
        processor = AutoProcessor.from_pretrained(base_path, local_files_only=True)
        processor.save_pretrained(staging)
        # vLLM resolves the image/video processors from their standalone
        # preprocessor config files even for text-only requests.  Preserve
        # these base-model files alongside the merged weights.
        for processor_file in ("preprocessor_config.json", "video_preprocessor_config.json"):
            source = base_path / processor_file
            if source.is_file():
                shutil.copy2(source, staging / processor_file)
        (staging / "merge_report.json").write_text(
            json.dumps(
                {
                    "checkpoint": Path(args.checkpoint).as_posix(),
                    "base_model": base_path.as_posix(),
                    "world_size": 1,
                    "rank": rank,
                    "lora_alpha": alpha,
                    "target_modules": targets,
                    "checkpoint_keys": len(state_dict),
                    "consumed_keys": used,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ) + "\n",
            encoding="utf-8",
        )

        # Release memory before the independent load verification.  This is
        # especially important on the single-GPU server.
        del peft, merged, base
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        verified_model = AutoModelForImageTextToText.from_pretrained(
            staging, torch_dtype=torch.bfloat16, local_files_only=True
        )
        verified_tokenizer = AutoTokenizer.from_pretrained(staging, local_files_only=True)
        verified_processor = AutoProcessor.from_pretrained(staging, local_files_only=True)
        del verified_model, verified_tokenizer, verified_processor, processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        staging.replace(output)
    except BaseException as exc:
        shutil.rmtree(staging, ignore_errors=True)
        print(f"error: merge verification failed: {exc}", file=sys.stderr)
        return 1

    print(f"[merge] verified merged HF model -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
