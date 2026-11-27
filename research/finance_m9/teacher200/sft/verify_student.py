"""Verify the finished checkpoint on CPU, including a fresh PEFT reload and forward pass."""
import hashlib
import json
from pathlib import Path
import sys


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    with path.open("rb") as source:
        return hashlib.sha256(source.read()).hexdigest()


def main(root, base):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    torch.set_num_threads(min(4, torch.get_num_threads()))
    folder = root / "planner-adapter"
    record = read(folder / "sft.json")
    cohort = read(Path(__file__).with_name("manifest.json"))
    count = cohort["rows"]
    updates = (count + 15) // 16
    if not (cohort["sources"] == 200 and cohort["preview"] is False and record["teacher"] == cohort
            and record["component"] == "planner" and record["updates"] == updates
            and digest(folder / "adapter_model.safetensors") == record["sha256"]
            and digest(Path(__file__).with_name("planner.jsonl")) == record["data_sha256"] == cohort["data_sha256"]):
        raise ValueError("Finished checkpoint and frozen dataset disagree")
    saved = torch.load(root / "resume.pt", map_location="cpu", weights_only=True)
    opt_steps = {int(s["step"]) for s in saved["optimizer"]["state"].values()}
    if saved["step"] != updates or opt_steps != {updates} or saved["identity"][:2] != (record["base_revision"], record["data_sha256"]):
        raise ValueError("Optimizer checkpoint does not attest every update")
    model = AutoModelForImageTextToText.from_pretrained(base, local_files_only=True, dtype=torch.bfloat16,
                                                        device_map={"": "cpu"}, attn_implementation="sdpa")
    model = PeftModel.from_pretrained(model, folder, is_trainable=False).eval()
    params = {name: value for name, value in model.named_parameters() if ".lora_" in name}
    if len(params) != 80 or sum(p.numel() for p in params.values()) != 3473408 or params.keys() != saved["adapter"].keys():
        raise ValueError("Reloaded adapter structure changed")
    for name, tensor in params.items():
        if not torch.isfinite(tensor).all() or not torch.equal(tensor, saved["adapter"][name]):
            raise ValueError(f"Reloaded weights differ from final optimizer step: {name}")
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    tokens = tokenizer("Plan the evidence checks for a database field.", return_tensors="pt").input_ids
    with torch.inference_mode():
        active = model(input_ids=tokens, use_cache=False, logits_to_keep=1).logits.float()
        with model.disable_adapter():
            plain = model(input_ids=tokens, use_cache=False, logits_to_keep=1).logits.float()
    change = (active - plain).abs().max().item()
    if not (torch.isfinite(active).all() and torch.isfinite(plain).all() and change > 0):
        raise ValueError("Reloaded adapter did not change finite logits")
    proof = dict(sources=200, examples=count, optimizer_updates=updates, reloaded_tensors=len(params),
                 reloaded_parameters=sum(p.numel() for p in params.values()), cpu_forward_finite=True,
                 max_logit_difference=change, sha256=record["sha256"], data_sha256=record["data_sha256"])
    tmp = root / "reload-check.partial"
    tmp.write_text(json.dumps(proof, indent=2))
    tmp.replace(root / "reload-check.json")
    print(json.dumps(proof), flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
