"""Pure helpers for an isolated, assistant-only semantic-Skill SFT pilot."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random

SEED = 20261001
SKILLS = ("SELECT_CATEGORY_TOPK", "INFER_CATEGORY_FROM_TOPK")
REVISION = "local-zip-003494285b21f01a5a6242cb45ae84394433fe72c41f447f64372720f78a2b20"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def validate_completed_adapter(adapter):
    adapter=Path(adapter)
    completion=read_json(adapter.parent/'completion.json')
    for file,key in [('adapter_model.safetensors','adapter_sha256'),('adapter_config.json','adapter_config_sha256')]:
        if sha256(adapter/file)!=completion[key]:
            raise ValueError(f'adapter {file} differs from completed training')


def validate_matched_baseline(base, adapted):
    if base['model']['adapter_sha256'] is not None:
        raise ValueError('control must use base weights without adapter')
    excluded={'adapter_sha256','adapter_config_sha256','adapter_routing'}
    if {k:v for k,v in base['model'].items() if k not in excluded} != {k:v for k,v in adapted['model'].items() if k not in excluded}:
        raise ValueError('matched model/generation parameters differ')
    if base['loop_config']!=adapted['loop_config']:
        raise ValueError('matched loop configuration differs')
    def behavior(manifest):
        return {k:v['sha256'] for k,v in manifest['files'].items() if k not in {'adapter_weights','adapter_config'}}
    if behavior(base)!=behavior(adapted):
        raise ValueError('matched non-adapter behavior artifacts differ')


def selected_skill(messages):
    """Route only trusted, exact Skill system headers, never user text."""
    if not messages or messages[0].get("role") != "system":
        return None
    system = messages[0].get("content", "")
    return next((name for name in SKILLS if system.startswith(f"Execute only the selected Skill {name}.\n")), None)


def category_of(row):
    levels = row["classification"]
    return str(row["domain"]) + ":" + "/".join(levels[k] for k in (
        "category_root_level", "category_branch_level", "category_leaf_level"
    ))


def auxiliary_candidates(gold, categories, source_id, seed=SEED):
    """Four reproducible distractors, explicitly NOT a gold semantic ranking."""
    rng = random.Random(f"{seed}:{source_id}")
    others = rng.sample(sorted(set(categories) - {gold}), 4)
    presented = [gold, *others]
    rng.shuffle(presented)
    return others, presented


def assistant_encoding(tokenizer, row, max_length=3072):
    """Mask prompts and weak candidate IDs; fail instead of truncating targets.

    Top-K has one true category only. IDs at positions 2..5 are scaffolding;
    any token overlapping these character spans receives label -100.
    """
    prompt = tokenizer.apply_chat_template(row["messages"], tokenize=False,
                                          add_generation_prompt=True, enable_thinking=False)
    response = row["response"]
    if not tokenizer.eos_token:
        raise ValueError("tokenizer requires an EOS token")
    text = prompt + response + tokenizer.eos_token
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    prefix = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    ids = encoded["input_ids"]
    if ids[:len(prefix)] != prefix:
        raise ValueError("generation prefix is not a token prefix of training sequence")
    if len(ids) > max_length:
        raise ValueError(f"{row['source_id']} exceeds max_length: {len(ids)} > {max_length}; no silent truncation")
    spans = [(len(prompt) + a, len(prompt) + b) for a, b in row["weak_response_spans"]]
    labels = []
    for index, (token, (a, b)) in enumerate(zip(ids, encoded["offset_mapping"])):
        ignored = index < len(prefix) or any(a < end and b > start for start, end in spans)
        labels.append(-100 if ignored else token)
    if not any(label != -100 for label in labels):
        raise ValueError("empty supervised completion")
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}, {
        "length": len(ids), "prompt_tokens": len(prefix),
        "supervised_tokens": sum(x != -100 for x in labels),
        "weak_tokens_masked": sum(index >= len(prefix) and label == -100 for index, label in enumerate(labels)),
    }
