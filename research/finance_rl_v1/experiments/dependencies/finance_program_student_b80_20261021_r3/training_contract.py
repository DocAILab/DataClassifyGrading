"""Standard-library-only, fail-closed contracts for the single B80 continuation."""
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time

BASE_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
ADAPTER_A_SHA256 = "81b18e85c5ce9ceb9585cf069a87d389567a2be47198c35a0d79d73f850a2abc"
ADAPTER_A_CONFIG_SHA256 = "df63e488cafe6b804c78a5f289e2d61d0ffae17893e8aebb9cfe6fe53f5bf757"
TOKENIZER_SHA256 = "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"
TRAIN_SHA256 = "2e72e51d1ae96894f96cf89f66d342b4a0f43876d7a7ab39915bd0c94e797f43"
MANIFEST_SHA256 = "d6e97dbda4da119924bebe7cbc5163051674ca64ec5a0f8f9040169eafb01f60"
MODEL_IDENTITY_SHA256 = "5b0a6b5e607d5305b2d59dd3e04bbd0f8b04af4502c389be3a5d83d01eddc8c9"
ORIGINAL_LORA_MODULES = frozenset(
    f"base_model.model.model.language_model.layers.{layer}.{module}"
    for layer in range(32)
    for module in (("self_attn.q_proj", "self_attn.v_proj") if layer % 4 == 3 else ("linear_attn.in_proj_qkv",))
)
ORIGINAL_LORA_NAMES = frozenset(f"{module}.lora_{side}.default.weight" for module in ORIGINAL_LORA_MODULES for side in "AB")


class ProtocolError(ValueError):
    pass


def sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def canonical_json_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class Dataset:
    rows: tuple
    train_sha256: str
    manifest_sha256: str
    manifest: dict


@dataclass(frozen=True)
class EncodedCompletion:
    input_ids: list
    attention_mask: list
    labels: list
    prompt_length: int
    target_length: int


def load_data(root, *, enforce_pinned=True):
    """Read only the locked train and manifest; never access local-only references.

    enforce_pinned=False is reserved for the explicit synthetic CPU runtime.
    Both normal callers and the production CLI use the pinned defaults.
    """
    paths = [Path(root) / "data" / name for name in ("train80.jsonl", "selection-manifest.json")]
    if any(p.is_symlink() or not p.is_file() for p in paths):
        raise ProtocolError("training inputs must be regular local files")
    raw, manifest_raw = [p.read_bytes() for p in paths]
    train_sha, manifest_sha = sha256_bytes(raw), sha256_bytes(manifest_raw)
    if enforce_pinned and (train_sha != TRAIN_SHA256 or manifest_sha != MANIFEST_SHA256):
        raise ProtocolError("B80 pinned data/manifest SHA mismatch")
    manifest = json.loads(manifest_raw)
    if manifest.get("outputs_sha256", {}).get("train80.jsonl") != train_sha:
        raise ProtocolError("manifest train byte SHA mismatch")
    lines = raw.splitlines()
    identities = [s for s in manifest["samples"] if s["split"] == "train"]
    if len(lines) != 80 or len(identities) != 80:
        raise ProtocolError("exactly 80 locked training rows required")
    rows = []
    for line, identity in zip(lines, identities):
        row = json.loads(line)
        if set(row) != {"source_id", "kind", "messages", "response", "trace_sha256"}:
            raise ProtocolError("training row must have exactly five fields")
        messages = row["messages"]
        if (not isinstance(messages, list) or [m.get("role") for m in messages] != ["system", "user"]
                or any(set(m) != {"role", "content"} or not isinstance(m["content"], str) for m in messages)):
            raise ProtocolError("exact system/user Planner messages required")
        if not isinstance(row["response"], str) or not row["response"]:
            raise ProtocolError("nonempty raw target required")
        if (row["source_id"] != identity["source_id"] or row["trace_sha256"] != identity["trace_sha256"]
                or sha256_bytes(canonical_json_bytes(messages)) != identity["messages_canonical_utf8_sha256"]
                or sha256_bytes(row["response"].encode("utf-8")) != identity["response_utf8_sha256"]):
            raise ProtocolError("row identity/order/messages/target SHA mismatch")
        rows.append(row)
    if len({r["source_id"] for r in rows}) != 80:
        raise ProtocolError("each B80 source must occur exactly once")
    return Dataset(tuple(rows), train_sha, manifest_sha, manifest)


def encode_completion_only(tokenizer, row, max_length=32768):
    if max_length != 32768:
        raise ProtocolError("fixed context length cannot be overridden")
    eos, eos_id = tokenizer.eos_token, tokenizer.eos_token_id
    if not isinstance(eos, str) or not eos or type(eos_id) is not int:
        raise ProtocolError("original EOS token and ID required")
    prompt = tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    prefix = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full = tokenizer(prompt + row["response"] + eos, add_special_tokens=False)["input_ids"]
    if not prefix or any(type(i) is not int for i in full + prefix) or full[:len(prefix)] != prefix:
        raise ProtocolError("prompt must be an exact integer-token prefix")
    target = full[len(prefix):]
    if len(full) > max_length:
        raise ProtocolError("context exceeds 32768; truncation forbidden")
    if len(target) < 2 or target[-1] != eos_id:
        raise ProtocolError("entire completion plus original EOS required")
    if tokenizer.decode(target, skip_special_tokens=False, clean_up_tokenization_spaces=False) != row["response"] + eos:
        raise ProtocolError("decoded target suffix differs from raw response plus EOS")
    return EncodedCompletion(full, [1] * len(full), [-100] * len(prefix) + target, len(prefix), len(target))


def validate_admission(admission, now=None):
    uuids = admission.get("gpu_uuids", [])
    if not isinstance(uuids, list) or len(uuids) not in (2, 3) or len(set(uuids)) != len(uuids) or any(not isinstance(u, str) or not u.startswith("GPU-") for u in uuids):
        raise ProtocolError("admission must lock two or three distinct GPU UUIDs")
    limits, memory = admission.get("allocator_limit_bytes", []), admission.get("model_max_memory_bytes", [])
    if len(limits) != len(uuids) or len(memory) != len(uuids):
        raise ProtocolError("admission memory vectors must match logical GPU count")
    if any(type(x) is not int or x <= 0 for x in limits + memory) or any(m > a for m, a in zip(memory, limits)):
        raise ProtocolError("positive model limits within allocator caps required")
    check_deadline(admission, now)


def check_deadline(admission, now=None):
    deadline = admission.get("gpu_deadline_unix")
    if type(deadline) not in (int, float) or not math.isfinite(deadline) or (time.time() if now is None else now) >= deadline:
        raise ProtocolError("absolute GPU deadline expired or invalid")


def verify_base_identity(model_path, identity_path=None):
    model_path = Path(model_path).resolve(strict=True)
    path = Path(identity_path) if identity_path else model_path / "model-identity.json"
    if not path.is_file() or path.is_symlink() or file_sha256(path) != MODEL_IDENTITY_SHA256:
        raise ProtocolError("original model-identity.json byte identity required")
    identity = json.loads(path.read_bytes())
    if identity["repo_id"] != "Qwen/Qwen3.5-9B" or identity["revision"] != BASE_REVISION or len(identity["files"]) != 11:
        raise ProtocolError("original pinned base identity mismatch")
    observed = {}
    for entry in identity["files"]:
        target = model_path / entry["path"]
        if target.parent != model_path or target.is_symlink() or not target.is_file():
            raise ProtocolError("base identity must refer to physical local regular files")
        actual = file_sha256(target)
        if target.stat().st_size != entry["bytes"] or actual != entry["sha256"]:
            raise ProtocolError("physical base file SHA mismatch: " + entry["path"])
        observed[entry["path"]] = actual
    config = json.loads((model_path / "config.json").read_bytes())
    if config.get("_commit_hash") not in (None, BASE_REVISION):
        raise ProtocolError("actual config revision contradicts pinned physical identity")
    return observed


def verify_tokenizer_identity(model_path, identity_path):
    """Bind rendering/tokenization metadata without opening any model weights."""
    model_path = Path(model_path).resolve(strict=True)
    identity_path = Path(identity_path)
    if identity_path.is_symlink() or not identity_path.is_file() or file_sha256(identity_path) != MODEL_IDENTITY_SHA256:
        raise ProtocolError("original model-identity.json byte identity required before tokenizer load")
    identity = json.loads(identity_path.read_bytes())
    if identity.get("repo_id") != "Qwen/Qwen3.5-9B" or identity.get("revision") != BASE_REVISION:
        raise ProtocolError("original tokenizer model/revision identity mismatch")
    dependencies = {"config.json", "tokenizer_config.json", "tokenizer.json", "chat_template.jinja", "merges.txt", "vocab.json"}
    entries = {entry["path"]: entry for entry in identity["files"]}
    if not dependencies.issubset(entries):
        raise ProtocolError("original identity is missing tokenizer metadata dependencies")
    # HF can pick up optional added-token/special-token files and alternate
    # templates. Reject such unbound inputs, including custom tokenizer code.
    for path in model_path.iterdir():
        name = path.name.lower()
        helper = (name.startswith(("tokenizer", "tokenization_", "configuration_", "chat_template", "added_token", "special_token", "vocab", "merges", "spiece", "sentencepiece"))
                  or name in {"tekken.json", "tokenizer.tiktoken"})
        if helper and path.name not in dependencies:
            raise ProtocolError("unexpected tokenizer helper/override: " + path.name)
    observed = {}
    for name in sorted(dependencies):
        path, entry = model_path / name, entries[name]
        if path.is_symlink() or not path.is_file() or path.stat().st_size != entry["bytes"]:
            raise ProtocolError("tokenizer dependency file/size mismatch: " + name)
        actual = file_sha256(path)
        if actual != entry["sha256"]:
            raise ProtocolError("tokenizer dependency SHA mismatch: " + name)
        observed[name] = actual
    return observed


def verify_adapter_identity(adapter_dir):
    adapter_dir = Path(adapter_dir).resolve(strict=True)
    result = {}
    for name, expected in (("adapter_model.safetensors", ADAPTER_A_SHA256), ("adapter_config.json", ADAPTER_A_CONFIG_SHA256)):
        path = adapter_dir / name
        if path.is_symlink() or not path.is_file() or file_sha256(path) != expected:
            raise ProtocolError("original A physical file SHA mismatch: " + name)
        result[name] = expected
    config = json.loads((adapter_dir / "adapter_config.json").read_bytes())
    if config.get("lora_dropout") != .05 or config.get("bias") != "none" or config.get("modules_to_save"):
        raise ProtocolError("original A dropout/adapter structure changed")
    return result


def validate_parameters(model, torch, *, cast=False):
    named = list(model.named_parameters())
    lora = [(n, p) for n, p in named if "lora_" in n]
    if len(lora) != 80 or {n for n, p in lora} != ORIGINAL_LORA_NAMES:
        raise ProtocolError("only exact original 80 language LoRA parameters permitted")
    for name, p in named:
        if name in ORIGINAL_LORA_NAMES:
            if cast:
                p.data = p.data.to(dtype=torch.float32)
                p.requires_grad_(True)
            if p.dtype != torch.float32 or not p.requires_grad or not torch.isfinite(p.detach()).all().item():
                raise ProtocolError("LoRA must be finite trainable FP32")
        elif p.dtype != torch.bfloat16 or p.requires_grad or p.grad is not None:
            raise ProtocolError("all base/vision parameters must stay frozen BF16")
    return lora


def verify_loaded_adapter(model, adapter_dir, torch):
    """Compare actual loaded FP32 values against all original safetensors values."""
    from safetensors.torch import load_file
    original = load_file(str(Path(adapter_dir) / "adapter_model.safetensors"), device="cpu")
    named = validate_parameters(model, torch)
    expected = {n.replace(".default.weight", ".weight"): p for n, p in named}
    if set(original) != set(expected):
        raise ProtocolError("original file tensor names differ from actual 80 parameters")
    for key, parameter in expected.items():
        if original[key].dtype != torch.float32 or not torch.equal(parameter.detach().float().cpu(), original[key]):
            raise ProtocolError("actually loaded original A values differ: " + key)
    return {"tensors": 80, "exact_values": True}
