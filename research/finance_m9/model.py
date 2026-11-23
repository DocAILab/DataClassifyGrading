"""Shared M9 model routing. Only a sampled Planner turn produces a policy action."""
from contextlib import contextmanager, nullcontext
import hashlib
from pathlib import Path

from hybrid_cache13 import prepare_options
from native_rank_input import make_rank_messages, rank_reply
from partial_contract import encode_option
from native_assets import ROOT, read
from rank_contract import compile_trie, rank_options
from resource_policy12 import generation_options
from routing import adapter_route

FRESH162 = "6d78e440ab0c9749427e002d54fd94cca06dea77c31b63ae3628b48fdbec4e64"
SCALE1169 = "0e52b9439b0016b0cad2903392030fca4b501019c7378a3a594b0fb6d1d4faf7"
TEMPERATURE = 0.8
BASE_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"


def sha(path):
    with Path(path).open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
        return digest.hexdigest()


def verify_adapter(directory, expected):
    if sha(Path(directory) / "adapter_model.safetensors") != expected:
        raise ValueError(f"Unexpected adapter weights in {directory}")


def sft_identity(directory, component, original):
    directory = Path(directory)
    digest = sha(directory / "adapter_model.safetensors")
    if digest != original:
        record = read(directory / "sft.json")
        if (record["component"], record["base_revision"], record["sha256"]) != (component, BASE_REVISION, digest):
            raise ValueError(f"Unexpected {component} SFT checkpoint")
    return digest


def load_base(directory):
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    for name, digest in read(ROOT / "assets/model-hashes.json").items():
        if sha(Path(directory) / name) != digest:
            raise ValueError(f"Unexpected Qwen3.5-9B file: {name}")
    tokenizer = AutoTokenizer.from_pretrained(directory, use_fast=True, local_files_only=True)
    base = AutoModelForImageTextToText.from_pretrained(
        directory, local_files_only=True, dtype=torch.bfloat16,
        device_map={"": 0}, attn_implementation="sdpa")
    return base, tokenizer


def load_backend(args, training=False):
    from peft import PeftModel

    planner_id = sft_identity(args.planner, "planner", FRESH162)
    skill_id = sft_identity(args.skill, "skill", SCALE1169)
    actor = args.planner
    if getattr(args, "candidate", None):
        actor = Path(args.candidate)
        record = read(actor.parent / "checkpoint.json")
        if (record["parent"], record["fixed_skill"]) != (planner_id, skill_id):
            raise ValueError("Checkpoint belongs to a different training protocol")
        verify_adapter(actor, record["planner_sha256"])
    base, tokenizer = load_base(args.model)
    model = PeftModel.from_pretrained(base, actor, adapter_name="planner", is_trainable=training)
    model.load_adapter(args.skill, adapter_name="skill_fixed", is_trainable=False)
    if training:
        model.load_adapter(args.planner, adapter_name="reference", is_trainable=False)
    for name in model.peft_config:
        config = model.peft_config[name]
        weights = [(key, p) for key, p in model.named_parameters() if ".lora_" in key and f".{name}." in key]
        if ((config.r, config.lora_alpha, config.bias) != (8, 16, "none") or len(weights) != 80
                or sum(p.numel() for _, p in weights) != 3473408
                or any(".language_model.layers." not in key for key, _ in weights)):
            raise ValueError(f"Unexpected LoRA target or scale for {name}")
    backend = Backend(model, tokenizer)
    backend.parent = planner_id
    backend.fixed_skill = skill_id
    return backend


class Backend:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.registry = read(ROOT / "registry-12.json")
        self.templates = read(ROOT / "prompt-templates.json")
        self.model.eval()
        self.model.requires_grad_(False)

    @contextmanager
    def route(self, name, *, learn=False):
        if learn and name != "planner":
            raise ValueError("Only the Planner adapter can be updated")
        if name is not None:
            self.model.set_adapter(name)
        ctx = self.model.disable_adapter() if name is None else nullcontext()
        try:
            with ctx:
                self.model.requires_grad_(False)
                if learn:
                    for param in self.planner_parameters():
                        param.requires_grad_(True)
                yield
        finally:
            # PEFT can re-enable the previous adapter when disable_adapter exits.
            self.model.requires_grad_(False)

    def planner_parameters(self):
        params = [p for n, p in self.model.named_parameters() if ".lora_" in n and ".planner." in n]
        if not params:
            raise ValueError("Planner LoRA tensors are missing")
        return params

    def tensor(self, ids):
        import torch
        return torch.tensor([ids], device=self.model.get_input_embeddings().weight.device)

    def rank(self, messages):
        import torch
        messages = make_rank_messages(messages, self.registry, self.templates)
        options = [entry["option_id"] for entry in self.registry]
        encoded = {i: encode_option(messages, i, self.tokenizer, 8192)[0]["input_ids"] for i in options}
        trie = compile_trie(encoded)
        if len(trie["nodes"]) != 3 or any(len(path) != 2 for path in trie["leaves"].values()):
            raise ValueError("Tokenizer does not match the frozen 12-ID search protocol")
        scores = {}
        with self.route(None), torch.inference_mode():
            for node in trie["nodes"]:
                x = self.tensor(node["input_ids"])
                logits = self.model(input_ids=x, attention_mask=torch.ones_like(x),
                                    use_cache=False, logits_to_keep=1).logits[0, -1].float()
                normalizer = torch.logsumexp(logits, dim=0)
                scores[node["node_id"]] = {str(i): float(logits[i] - normalizer) for i in node["next_tokens"]}
        return rank_reply(rank_options(trie, scores, options)["option_ids"])

    def call(self, stage, messages, *, sample):
        import torch
        if stage.startswith("SEARCH_STANDARD_"):
            return self.rank(messages), None
        route = adapter_route(stage)
        tok = self.tokenizer
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        prompt = tok(text, add_special_tokens=False)["input_ids"]
        options = prepare_options(self.model, generation_options(len(prompt), 1536 if stage == "Planner" else 768))
        options["pad_token_id"] = tok.eos_token_id
        stochastic = sample and stage == "Planner"
        if stochastic:
            options.update(do_sample=True, top_k=0, top_p=1., temperature=TEMPERATURE,
                           repetition_penalty=1., return_dict_in_generate=True, output_scores=True)
        x = self.tensor(prompt)
        with self.route(route), torch.inference_mode():
            output = self.model.generate(input_ids=x, attention_mask=torch.ones_like(x), **options)
            seq = output.sequences if stochastic else output
            tokens = seq[0, len(prompt):].tolist()
            if not tokens or tokens[-1] not in options["eos_token_id"]:
                raise ValueError(f"{stage} generation ended without EOS")
            action = None
            if stochastic:
                # Generation scores already include the sampling temperature.
                behavior = [float(s[0].float().log_softmax(-1)[i])
                            for s, i in zip(output.scores, tokens, strict=True)]
                action = {"prompt": prompt, "tokens": tokens, "behavior": behavior}
        return tok.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False), action
