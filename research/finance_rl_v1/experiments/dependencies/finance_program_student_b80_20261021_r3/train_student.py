"""One durable B80 continuation. No retries, model downloads or hyperparameter CLI."""
import argparse
from contextlib import nullcontext
import importlib
import json
import math
import os
from pathlib import Path
import time
from memory_runtime import configure_gradient_checkpointing

from training_contract import (
    BASE_REVISION, TOKENIZER_SHA256, ProtocolError,
    canonical_json_bytes, sha256_bytes, file_sha256, load_data,
    encode_completion_only, validate_admission, check_deadline,
    verify_base_identity, verify_tokenizer_identity, verify_adapter_identity, validate_parameters,
    verify_loaded_adapter,
)


def exclusive_json(path, value):
    with Path(path).open("xb") as handle:
        handle.write(canonical_json_bytes(value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def _load_tokenizer(model_path):
    if file_sha256(Path(model_path) / "tokenizer.json") != TOKENIZER_SHA256:
        raise ProtocolError("original tokenizer.json SHA mismatch")
    transformers = importlib.import_module("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, use_fast=True)
    if not tokenizer.is_fast:
        raise ProtocolError("original local fast tokenizer required")
    return tokenizer


def _token_evidence(dataset, tokenizer):
    encoded = [encode_completion_only(tokenizer, row) for row in dataset.rows]
    details = [dict(source_id=row["source_id"], input_tokens=len(enc.input_ids), prompt_tokens=enc.prompt_length,
                    supervised_tokens=enc.target_length, boundary_exact=True, decoded_suffix_exact=True,
                    target_ids_sha256=sha256_bytes(canonical_json_bytes(enc.input_ids[enc.prompt_length:])), truncated=False)
               for row, enc in zip(dataset.rows, encoded)]
    return encoded, details


def audit_tokenizer(root, model_path, output):
    """CPU-only entry: imports tokenizer classes, never torch/models/CUDA."""
    dataset = load_data(root)
    identity_path = Path(root) / "model-identity.json"
    tokenizer_files = verify_tokenizer_identity(model_path, identity_path)
    tokenizer = _load_tokenizer(model_path)
    _, rows = _token_evidence(dataset, tokenizer)
    result = dict(schema="b80-tokenizer-audit-v1", rows=80, train_sha256=dataset.train_sha256,
                  manifest_sha256=dataset.manifest_sha256, tokenizer_json_sha256=TOKENIZER_SHA256,
                  tokenizer_config_sha256=file_sha256(Path(model_path) / "tokenizer_config.json"),
                  tokenizer_files_sha256=tokenizer_files, model_identity_sha256=file_sha256(identity_path),
                  eos_token_id=tokenizer.eos_token_id, enable_thinking=False, max_length=32768,
                  max_input_tokens=max(r["input_tokens"] for r in rows), per_source=rows)
    exclusive_json(output, result)
    return result


def completion_loss(model, encoded, torch):
    """FP32 CE of the exact completion suffix, preserving example-mean weighting."""
    count, start = encoded.target_length, encoded.prompt_length
    if count < 2 or start < 1 or encoded.labels != [-100] * start + encoded.input_ids[start:]:
        raise ProtocolError("completion-only contiguous mask mismatch")
    device = model.get_input_embeddings().weight.device
    ids = torch.tensor([encoded.input_ids], dtype=torch.long, device=device)
    attention = torch.tensor([encoded.attention_mask], dtype=torch.long, device=device)
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
    with autocast:
        output = model(input_ids=ids, attention_mask=attention, use_cache=False, logits_to_keep=count + 1)
    logits = output.logits
    if logits.ndim != 3 or logits.shape[:2] != (1, count + 1):
        raise ProtocolError("model must return target_count+1 logits")
    target = torch.tensor(encoded.labels[start:], dtype=torch.long, device=logits.device)
    loss = torch.nn.functional.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]), target)
    if not torch.isfinite(loss.detach()).item():
        raise FloatingPointError("nonfinite completion loss")
    return loss


def global_clip(named, torch):
    grads = [p.grad for _, p in named]
    if any(g is None or g.dtype != torch.float32 or not torch.isfinite(g).all().item() for g in grads):
        raise ProtocolError("all 80 gradients must be finite FP32")
    # Combine every device's contribution before applying the same coefficient.
    norm = math.sqrt(sum(float(g.detach().double().square().sum().item()) for g in grads))
    if not math.isfinite(norm) or norm <= 0:
        raise ProtocolError("finite nonzero global gradient required")
    coefficient = min(1., 1. / (norm + 1e-6))
    for g in grads:
        g.mul_(coefficient)
    return norm, norm * coefficient


def memory_snapshot(torch, model, cpu_test=False):
    if cpu_test:
        return {"cpu_test_only": True, "hf_device_map": {}, "devices": []}
    devices = []
    for index in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(index)
        devices.append(dict(logical_device=index, allocated=torch.cuda.memory_allocated(index),
                            reserved=torch.cuda.memory_reserved(index), peak_allocated=torch.cuda.max_memory_allocated(index),
                            peak_reserved=torch.cuda.max_memory_reserved(index), free=free, total=total))
    return {"hf_device_map": {str(k): str(v) for k, v in model.hf_device_map.items()}, "devices": devices}


def _runtime_loader(root, admission, tokenizer):
    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")
    peft = importlib.import_module("peft")
    count = len(admission["gpu_uuids"])
    if os.environ.get("CUDA_VISIBLE_DEVICES") != ",".join(admission["gpu_uuids"]):
        raise ProtocolError("CUDA_VISIBLE_DEVICES must equal admitted UUIDs in order")
    if not torch.cuda.is_available() or torch.cuda.device_count() != count:
        raise ProtocolError("visible CUDA count differs from admission")
    for index, cap in enumerate(admission["allocator_limit_bytes"]):
        with torch.cuda.device(index):
            if not torch.cuda.is_bf16_supported():
                raise ProtocolError("each admitted device must support BF16")
            # PyTorch _CUuuid omits the nvidia-smi GPU- prefix. Canonicalize
            # only the UUID representation; selected physical identity stays exact.
            from uuid import UUID
            observed = 'GPU-' + str(UUID(str(torch.cuda.get_device_properties(index).uuid).removeprefix('GPU-')))
            if observed != admission["gpu_uuids"][index]:
                raise ProtocolError("actual visible GPU UUID order differs from admission")
            total = torch.cuda.get_device_properties(index).total_memory
            if cap > total:
                raise ProtocolError("allocator cap exceeds actual GPU memory")
            torch.cuda.set_per_process_memory_fraction(cap / total, index)
    transformers.set_seed(20261021)
    check_deadline(admission)
    base = transformers.AutoModelForImageTextToText.from_pretrained(
        str(admission["base_model_dir"]), local_files_only=True, dtype=torch.bfloat16,
        attn_implementation="sdpa", device_map="balanced",
        max_memory=dict(enumerate(admission["model_max_memory_bytes"])))
    if getattr(base.config, "_commit_hash", None) not in (None, BASE_REVISION):
        raise ProtocolError("loaded config contradicts validated physical base revision")
    placements = {str(v).replace("cuda:", "") for v in base.hf_device_map.values()}
    if placements != {str(i) for i in range(count)}:
        raise ProtocolError("model must occupy all admitted GPUs, with no CPU/disk placement")
    model = peft.PeftModel.from_pretrained(base, str(admission["original_adapter_dir"]), is_trainable=True, local_files_only=True)
    if set(model.peft_config) != {"default"} or model.active_adapter not in ("default", ["default"], ("default",)):
        raise ProtocolError("exact original default A adapter required")
    if model.peft_config["default"].lora_dropout != .05:
        raise ProtocolError("original dropout must remain .05")
    model.config.use_cache = False
    return dict(torch=torch, transformers=transformers, peft=peft, model=model, tokenizer=tokenizer)


def _claim(root, original=None):
    root = Path(root).absolute()
    if root.resolve() != root or root.is_symlink():
        raise ProtocolError("output root must be physical, without symlink components")
    parent = root / "runs"
    if parent.is_symlink() or (parent.exists() and parent.resolve() != parent):
        raise ProtocolError("runs parent may not alias another directory")
    parent.mkdir(exist_ok=True)
    run = parent / "training"
    if original is not None and (run == original or original in run.parents):
        raise ProtocolError("output may not be under original adapter")
    if run.is_symlink() or run.exists():
        raise ProtocolError("preexisting training run refused; no resume/retry")
    run.mkdir(exist_ok=False)
    exclusive_json(run / "claim.json", {"pid": os.getpid(), "time_unix": time.time()})
    return run


def _save_final(model, destination, runtime):
    destination.mkdir(exist_ok=False)
    if runtime.get("cpu_test_only") is True:
        runtime["save_adapter"](model, destination)
        return {p.name: file_sha256(p) for p in destination.iterdir()}
    model.save_pretrained(str(destination), safe_serialization=True, selected_adapters=["default"])
    card = destination / "README.md"
    auxiliary = []
    if card.exists():
        if card.is_symlink() or not card.is_file():
            raise ProtocolError("PEFT README must be a physical file")
        auxdir = destination.parent / "auxiliary"
        auxdir.mkdir(exist_ok=False)
        with (auxdir / "README.md").open("xb") as handle:
            handle.write(card.read_bytes())
            handle.flush()
            os.fsync(handle.fileno())
        auxiliary.append({"path": "auxiliary/README.md", "sha256": file_sha256(card)})
        card.unlink()
    expected = {"adapter_config.json", "adapter_model.safetensors"}
    if {p.name for p in destination.iterdir()} != expected or any(p.is_symlink() or not p.is_file() for p in destination.iterdir()):
        raise ProtocolError("final adapter must contain exactly config and safetensors")
    # Saved file must contain all and only the actual updated 80 tensors.
    verify_loaded_adapter(model, destination, runtime["torch"])
    for p in destination.iterdir():
        with p.open("rb") as handle:
            os.fsync(handle.fileno())
    return {"files_sha256": {p.name: file_sha256(p) for p in destination.iterdir()}, "auxiliary": auxiliary,
            "tensor_count": 80, "actual_values_exact": True}


def run_training(root, admission, runtime=None):
    """Execute exactly 80 ordered microbatches and 10 actual AdamW steps.

    runtime injection is only accepted when cpu_test_only is explicitly True;
    no CLI exposes this test facility. The same loop and autograd serve both.
    """
    root = Path(root)
    validate_admission(admission)
    test = runtime is not None
    if test and runtime.get("cpu_test_only") is not True:
        raise ProtocolError("runtime injection requires explicit synthetic CPU test mode")
    dataset = load_data(root, enforce_pinned=not test)
    identities = None
    original = None
    if not test:
        original = Path(admission["original_adapter_dir"]).resolve(strict=True)
        identities = {"base": verify_base_identity(admission["base_model_dir"], root / "model-identity.json"),
                      "adapter": verify_adapter_identity(original)}
        verify_tokenizer_identity(admission["base_model_dir"], root / "model-identity.json")
        tokenizer = _load_tokenizer(admission["base_model_dir"])
        encoded, token_rows = _token_evidence(dataset, tokenizer)
    else:
        if any(p.device.type != "cpu" for _, p in runtime["model"].named_parameters()):
            raise ProtocolError("synthetic runtime parameters must all reside on CPU")
        tokenizer = runtime["tokenizer"]
        encoded, token_rows = _token_evidence(dataset, tokenizer)
    run = _claim(root, original)
    if not test:
        exclusive_json(run / "preload-identities.json", identities)
        runtime = _runtime_loader(root, admission, tokenizer)
    torch, model = runtime["torch"], runtime["model"]
    torch.manual_seed(20261021)
    model.config.use_cache = False
    named = validate_parameters(model, torch, cast=True)
    if not test:
        if {p.device.index for _, p in model.named_parameters() if p.device.type == "cuda"} != set(range(len(admission["gpu_uuids"]))) or any(p.device.type != "cuda" for _, p in model.named_parameters()):
            raise ProtocolError("actual model parameters must occupy every admitted GPU without CPU/disk offload")
        initial = verify_loaded_adapter(model, original, torch)
        exclusive_json(run / "initial-loaded-adapter.json", initial)
    if test:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
    else:
        checkpoint_configuration = configure_gradient_checkpointing(model)
        exclusive_json(run / "checkpoint-configuration.json", checkpoint_configuration)
    model.enable_input_require_grads()
    model.train()
    optimizer_cls = runtime.get("optimizer_factory", torch.optim.AdamW) if test else torch.optim.AdamW
    optimizer = optimizer_cls([p for _, p in named], lr=1e-5, betas=(.9, .999), eps=1e-8, weight_decay=.01)
    consumed, evidence = [], []
    start_memory = memory_snapshot(torch, model, test)
    exclusive_json(run / "memory-start.json", start_memory)
    for step in range(1, 11):
        check_deadline(admission)
        optimizer.zero_grad(set_to_none=True)
        before = {n: p.detach().cpu().clone() for n, p in named}
        losses = []
        for index in range((step - 1) * 8, step * 8):
            check_deadline(admission)
            row = dataset.rows[index]
            phase = dict(step=step, micro=index + 1, source_id=row["source_id"], time_unix=time.time())
            prefix = f"micro-{index + 1:03d}"
            exclusive_json(run / f"{prefix}.forward-start.json", {**phase, "phase": "forward-start"})
            with torch.autograd.graph.save_on_cpu(pin_memory=not test):
                loss = completion_loss(model, encoded[index], torch)
                losses.append(float(loss.detach().item()))
                exclusive_json(run / f"{prefix}.forward-return.json", {**phase, "phase": "forward-return", "loss": losses[-1], "time_unix": time.time()})
                (loss / 8).backward()
                exclusive_json(run / f"{prefix}.backward-return.json", {**phase, "phase": "backward-return", "time_unix": time.time()})
            consumed.append(row["source_id"])
        grad_norm, clipped_norm = global_clip(named, torch)
        check_deadline(admission)
        exclusive_json(run / f"step-{step:02d}.optimizer-start.json", dict(step=step, phase="optimizer-start", time_unix=time.time()))
        optimizer.step()
        exclusive_json(run / f"step-{step:02d}.optimizer-return.json", dict(step=step, phase="optimizer-return", time_unix=time.time()))
        state_count = 0
        for _, p in named:
            state = optimizer.state[p]
            if not {"exp_avg", "exp_avg_sq", "step"}.issubset(state):
                raise ProtocolError("actual AdamW moments/step missing")
            for value in state.values():
                if torch.is_tensor(value):
                    state_count += 1
                    if value.dtype != torch.float32 or not torch.isfinite(value).all().item():
                        raise ProtocolError("all floating optimizer states must be finite FP32")
            if float(state["step"].item()) != step:
                raise ProtocolError("AdamW actual step accounting mismatch")
        delta = math.sqrt(sum(float((p.detach().cpu() - before[n]).double().square().sum().item()) for n, p in named))
        if not math.isfinite(delta) or delta <= 0:
            raise ProtocolError("optimizer step must produce a finite positive actual delta")
        validate_parameters(model, torch)
        item = dict(step=step, loss=sum(losses) / 8, micro_losses=losses, grad_norm=grad_norm,
                    clipped_grad_norm=clipped_norm, update_norm=delta, optimizer_state_tensor_count=state_count,
                    rows_consumed=consumed[-8:], completed=True, memory=memory_snapshot(torch, model, test))
        exclusive_json(run / f"step-{step:02d}.evidence.json", item)
        evidence.append(item)
    if consumed != [r["source_id"] for r in dataset.rows] or len(set(consumed)) != 80:
        raise ProtocolError("80 original sources must be consumed once in locked order")
    check_deadline(admission)
    destination = run / "final-adapter"
    if original is not None and (destination.resolve() == original or original in destination.resolve().parents):
        raise ProtocolError("final adapter aliases original A")
    final = _save_final(model, destination, runtime)
    validate_parameters(model, torch)
    if not test:
        if verify_base_identity(admission["base_model_dir"], root / "model-identity.json") != identities["base"] or verify_adapter_identity(original) != identities["adapter"]:
            raise ProtocolError("base or original A changed during training/save")
    check_deadline(admission)
    result = dict(schema="b80-single-continuation-v1", cpu_test_only=test, train_sha256=dataset.train_sha256,
                  manifest_sha256=dataset.manifest_sha256, rows_consumed=consumed, rows=80, epochs=1,
                  micro_batch_size=1, accumulation=8, learning_rate=1e-5, seed=20261021,
                  optimizer_steps=evidence, trainable_names=sorted(n for n, _ in named),
                  encoded_rows=token_rows, final_adapter=final, model_identities=identities,
                  versions={name: str(getattr(runtime.get(name), "__version__", "synthetic-cpu")) for name in ("torch", "transformers", "peft")},
                  memory=memory_snapshot(torch, model, test), admission=admission)
    exclusive_json(run / "completion.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--audit-tokenizer", action="store_true")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--admission", type=Path)
    args = parser.parse_args(argv)
    if args.audit_tokenizer:
        if args.model_path is None or args.output is None or args.admission is not None:
            parser.error("tokenizer audit requires --model-path and --output only")
        result = audit_tokenizer(args.root, args.model_path, args.output)
    else:
        if args.admission is None or args.model_path is not None or args.output is not None:
            parser.error("training requires the single locked --admission JSON")
        result = run_training(args.root, json.loads(args.admission.read_bytes()))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

