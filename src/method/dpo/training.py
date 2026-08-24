"""Model loading, runtime guards, and metrics for Stage 2 DPO."""

from __future__ import annotations

from importlib import metadata
import math
from pathlib import Path
from typing import Callable, Mapping


DPO_DEFAULTS = {
    "beta": 0.1,
    "learning_rate": 5e-7,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "num_train_epochs": 1.0,
    "bf16": True,
    "gradient_checkpointing": True,
    "save_steps": 100,
    "seed": 42,
}

REQUIRED_VERSIONS = {
    "trl": "0.24.0",
    "peft": "0.20.0",
    "transformers": "5.15.0",
    "datasets": "5.0.1",
}


def validate_runtime_versions(
    resolver: Callable[[str], str] = metadata.version,
) -> dict[str, str]:
    """Require the exact server stack that was inspected for this experiment."""
    actual = {name: resolver(name) for name in REQUIRED_VERSIONS}
    mismatches = {
        name: (REQUIRED_VERSIONS[name], version)
        for name, version in actual.items()
        if version != REQUIRED_VERSIONS[name]
    }
    if mismatches:
        details = ", ".join(
            f"{name}: expected {expected}, got {got}"
            for name, (expected, got) in mismatches.items()
        )
        raise RuntimeError(f"DPO runtime version mismatch: {details}")
    return actual


def normalize_trl_availability_flags(module) -> dict[str, bool]:
    """Work around TRL 0.24 tuple-valued optional-dependency flags."""
    changed: dict[str, bool] = {}
    for name, value in list(vars(module).items()):
        if name.startswith("_") and name.endswith("_available") and isinstance(value, tuple):
            normalized = bool(value[0])
            setattr(module, name, normalized)
            changed[name] = normalized
    return dict(sorted(changed.items()))


def assert_separate_output(sft_adapter: str | Path, output_dir: str | Path) -> None:
    """Refuse to overwrite the immutable public SFT adapter."""
    sft = Path(sft_adapter).resolve()
    output = Path(output_dir).resolve()
    if output == sft or sft in output.parents:
        raise ValueError("DPO output must be separate from the immutable SFT adapter")


def freeze_reference(model) -> None:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def parameter_update_stats(before: Mapping[str, object], after: Mapping[str, object]) -> dict[str, float]:
    """Compute absolute and relative L2 update norms for matching tensors."""
    import torch

    if set(before) != set(after):
        raise ValueError("parameter names differ between snapshots")
    update_sq = 0.0
    base_sq = 0.0
    for name in before:
        left = torch.as_tensor(before[name]).detach().float().cpu()
        right = torch.as_tensor(after[name]).detach().float().cpu()
        if left.shape != right.shape:
            raise ValueError(f"parameter shape changed: {name}")
        update_sq += float(torch.sum((right - left) ** 2))
        base_sq += float(torch.sum(left ** 2))
    absolute = math.sqrt(update_sq)
    relative = absolute / math.sqrt(base_sq) if base_sq else (math.inf if absolute else 0.0)
    return {"absolute_update_norm": absolute, "relative_update_norm": relative}


def snapshot_trainable_parameters(model) -> dict[str, object]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def model_parameter_fingerprint(model) -> dict[str, float]:
    """Small constant-memory fingerprint suitable for frozen-reference checks."""
    import torch

    total_sum = 0.0
    total_sq = 0.0
    count = 0
    with torch.no_grad():
        for parameter in model.parameters():
            value = parameter.detach().float()
            total_sum += float(value.sum())
            total_sq += float(torch.sum(value * value))
            count += value.numel()
    return {"sum": total_sum, "l2": math.sqrt(total_sq), "count": float(count)}


def load_merged_sft_model(
    model_path: str | Path,
    sft_adapter: str | Path,
    *,
    device_map: str = "cuda",
):
    """Load base+SFT and merge in memory without writing a 15GB model."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        local_files_only=True,
        attn_implementation="sdpa",
    )
    return PeftModel.from_pretrained(
        base, sft_adapter, is_trainable=False
    ).merge_and_unload(safe_merge=True)


def create_dpo_trainer(
    policy,
    reference,
    tokenizer,
    train_dataset,
    output_dir: str | Path,
    *,
    max_length: int = 512,
    max_prompt_length: int = 384,
    max_steps: int = -1,
):
    """Create a TRL trainer after normalizing its broken optional flags."""
    import trl.import_utils as trl_import_utils

    normalize_trl_availability_flags(trl_import_utils)
    from peft import LoraConfig
    from trl import DPOConfig, DPOTrainer

    freeze_reference(reference)
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    kwargs = dict(DPO_DEFAULTS)
    args = DPOConfig(
        output_dir=str(output_dir),
        **kwargs,
        max_steps=max_steps,
        max_length=max_length,
        max_prompt_length=max_prompt_length,
        logging_steps=1,
        save_strategy="steps",
        save_total_limit=2,
        report_to=[],
        eval_strategy="no",
        remove_unused_columns=False,
        loss_type="sigmoid",
    )
    return DPOTrainer(
        model=policy,
        ref_model=reference,
        args=args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
