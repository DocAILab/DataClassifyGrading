"""Train either of the two M9 adapters from the original Qwen base."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile

from batch_plan import make_batches
from model import BASE_REVISION, load_base, sha
from partial_contract import make_decision, encode_option
from native_assets import ROOT, EXPERIMENTS, read


def planner_examples(path, tok):
    import importlib.util
    if sha(path) != "64485902717aa33ce58f5ed0719ed22f49b3c836b8739ef0cf45f42f47761020":
        raise ValueError("Expected the frozen fresh162 train162.jsonl in its original order and encoding")
    source = EXPERIMENTS / "dependencies/finance_program_student_fresh162_20261022_r5/dataset_contract.py"
    spec = importlib.util.spec_from_file_location("fresh162_dataset", source)
    contract = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(contract)
    rows = [json.loads(line) for line in path.read_bytes().splitlines()]
    if len(rows) != 162 or len({r["id"] for r in rows}) != 162:
        raise ValueError("Supply the original ordered 162 demonstrations")
    examples = []
    for row in rows:
        if set(row) != {"id", "messages", "completion"} or [m["role"] for m in row["messages"]] != ["system", "user"]:
            raise ValueError("Invalid Planner demonstration")
        contract.validate_program(row["completion"])
        encoded = contract.encode_completion(tok, row, 49152)
        examples.append((encoded["input_ids"], encoded["target_length"]))
    return examples


def skill_examples(path, tok):
    data = read(path)
    rows = data["rows"]
    if data["split"] != "TRAIN" or len(rows) != 1169 or len({r["source_id"] for r in rows}) != 1169:
        raise ValueError("Scale1169 requires 1169 distinct TRAIN sources")
    payload = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    if hashlib.sha256(payload).hexdigest() != "5e9b592d51fa5b569916b58203737959e4fb249cb16a9e0afed9aee253406d0f":
        raise ValueError("Scale1169 source order, metadata or targets differ from the training cohort")
    registry, templates = read(ROOT / "registry-12.json"), read(ROOT / "prompt-templates.json")
    ids = [item["option_id"] for item in registry]
    examples = []
    for row in rows:
        if set(row) != {"metadata", "target_option_id", "source_id"}:
            raise ValueError("Unexpected training input keys")
        messages = make_decision({"metadata": row["metadata"]}, registry, templates, ids)
        encoded, audit = encode_option(messages, row["target_option_id"], tok, 4096)
        examples.append((encoded["input_ids"], audit["supervised_tokens"]))
    return examples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("component", choices=["planner", "skill"])
    for key in ("model", "data", "output"):
        parser.add_argument("--" + key, required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output directory")
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import set_seed

    planner = args.component == "planner"
    seed = 20261022 if planner else 2026110502
    set_seed(seed)
    base, tok = load_base(args.model)
    targets = [n for n, m in base.named_modules() if n.startswith("model.language_model.layers.")
               and n.endswith((".q_proj", ".v_proj", ".in_proj_qkv")) and isinstance(m, torch.nn.Linear)]
    if len(targets) != 40:
        raise ValueError("Unexpected language backbone")
    examples = (planner_examples if planner else skill_examples)(args.data, tok)
    model = get_peft_model(base, LoraConfig(r=8, lora_alpha=16, lora_dropout=.05,
                           bias="none", task_type="CAUSAL_LM", target_modules=targets))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.train()
    active = [p for p in model.parameters() if p.requires_grad]
    if sum(p.numel() for p in active) != 3473408:
        raise ValueError("Unexpected trainable parameters")
    optimizer = torch.optim.AdamW(active, lr=1e-4 if planner else 5e-5, weight_decay=.01)
    batches = ([list(range(i, min(i + 16, len(examples)))) for i in range(0, len(examples), 16)]
               if planner else make_batches(len(examples), 8, seed))
    for step, batch in enumerate(batches, 1):
        optimizer.zero_grad(set_to_none=True)
        for index in batch:
            tokens, n = examples[index]
            x = torch.tensor([tokens], device=model.get_input_embeddings().weight.device)
            logits = model(input_ids=x, attention_mask=torch.ones_like(x),
                           use_cache=False, logits_to_keep=n + 1).logits[0, :-1].float()
            loss = torch.nn.functional.cross_entropy(logits, x[0, -n:])
            (loss / len(batch)).backward()
        torch.nn.utils.clip_grad_norm_(active, 1., error_if_nonfinite=True)
        optimizer.step()
        print(json.dumps({"step": step, "examples": len(batch)}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output.parent) as tmp:
        saved = Path(tmp) / "adapter"
        model.save_pretrained(saved)
        record = {"component": args.component, "base_revision": BASE_REVISION,
                  "data_sha256": sha(args.data), "updates": len(batches),
                  "sha256": sha(saved / "adapter_model.safetensors")}
        (saved / "sft.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        saved.rename(args.output)


if __name__ == "__main__":
    main()
