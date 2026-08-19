"""Merge a verl LoRA FSDP checkpoint into a standalone HF model directory.

verl saves LoRA checkpoints in peft-compatible state-dict layout
(``base_model.model....lora_A.default.weight`` + ``lora_train_meta.json``).
This script rebuilds a PeftModel on the base weights, loads the checkpoint
state dict, merges LoRA into the base and saves a plain HF directory that any
evaluator / downstream RL init can load normally (no verl dependency).

Verification: reports how many checkpoint keys were consumed; a LoRA merge
with the frozen base weights must consume 100% of keys (base + lora).

Usage:
  python -m script.verl.sft.merge_lora_checkpoint \
    --checkpoint <verl global_step dir> \
    --base-model <HF base dir> \
    --output <merged HF dir>
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import sys

import torch

LORA_A = re.compile(r"(.*)\.lora_A\.default\.weight$")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="verl checkpoint dir (has model_world_size_1_rank_0.pt)")
    parser.add_argument("--base-model", required=True, help="HF base model dir (same as trained)")
    parser.add_argument("--output", required=True, help="merged HF output dir")
    args = parser.parse_args(argv)

    ckpt_dir = Path(args.checkpoint)
    model_file = ckpt_dir / "model_world_size_1_rank_0.pt"
    meta_file = ckpt_dir / "lora_train_meta.json"
    if not model_file.is_file():
        print(f"error: {model_file} not found", file=sys.stderr)
        return 2

    sd = torch.load(model_file, map_location="cpu", weights_only=False)
    keys = list(sd.keys())
    targets = sorted(
        {m.group(1).rsplit(".", 1)[-1] for k in keys for m in [LORA_A.match(k)] if m}
    )
    if not targets:
        print("error: no LoRA keys found in checkpoint", file=sys.stderr)
        return 2

    meta = {}
    if meta_file.is_file():
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
    rank = meta.get("r", 8)
    alpha = meta.get("lora_alpha", rank)

    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[merge] targets: {targets}")
    print(f"[merge] r={rank} alpha={alpha} keys={len(keys)}")
    base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16)
    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=targets,
        task_type="CAUSAL_LM",
    )
    peft = PeftModel(base, config)
    missing, unexpected = peft.load_state_dict(sd, strict=False)
    unexpected = [k for k in unexpected if not k.startswith("base_model.model.model")]
    if unexpected:
        print(f"warning: {len(unexpected)} unexpected keys (ignored): {unexpected[:5]}")
    used = len(keys) - len(unexpected)
    ratio = used / len(keys)
    print(f"[merge] consumed {used}/{len(keys)} checkpoint keys ({ratio:.1%})")
    if ratio < 0.99:
        print("error: checkpoint keys consumed < 99%; aborting", file=sys.stderr)
        return 1
    merged = peft.merge_and_unload()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.save_pretrained(out)
    print(f"[merge] saved merged model -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
