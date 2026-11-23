"""On-policy Planner GRPO with a frozen fresh162 reference and M9 Skill routing."""
import importlib.util
import json
from pathlib import Path
import tempfile

from episode import run_episode
from model import TEMPERATURE, sha
from native_assets import EXPERIMENTS, read

MATH = EXPERIMENTS / "finance_program_grpo_full_migrated_20261027_r3"


def _math_module(name):
    spec = importlib.util.spec_from_file_location("m9_" + name, MATH / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


policy = _math_module("policy_math")
advantages = _math_module("grpo_math").advantages


def references(path, public, split):
    data = read(path)
    refs = data["references"]
    study = public["inputs"]["study"]
    if data["split"] != split or set(refs) != {r["id"] for r in study["rows"]}:
        raise ValueError("Reference split and source IDs must exactly match the input")
    for ref in refs.values():
        if set(ref) != {"category", "grading_level"} or ref["category"] not in study["registry"]:
            raise ValueError("Expected a canonical finance category and grading_level")
        if ref["grading_level"] not in {v["grading_level"] for v in study["registry"].values()}:
            raise ValueError("Unknown reference grade")
    return refs


def update(backend, optimizer, groups):
    import torch
    rewards = [[reward for reward, actions in group] for group in groups]
    normalized = advantages(rewards, rewards)["advantages"]
    turns = [(action, advantage) for group, weights in zip(groups, normalized, strict=True)
             for (_, actions), advantage in zip(group, weights, strict=True) if advantage
             for action in actions]
    if not turns:
        return False
    count = sum(len(action["tokens"]) for action, _ in turns)
    optimizer.zero_grad(set_to_none=True)
    for action, advantage in turns:
        tokens = (action["prompt"], action["tokens"])
        with backend.route("reference"), torch.no_grad():
            ref = policy.action_log_probs(backend.model, *tokens, temperature=TEMPERATURE)
        with backend.route("planner", learn=True), torch.autograd.graph.save_on_cpu():
            current = policy.action_log_probs(backend.model, *tokens, temperature=TEMPERATURE)
            # All groups in this batch were sampled before any optimizer step.
            loss = policy.corrected_token_objective(
                current, current.detach(), torch.tensor(action["behavior"]), ref, advantage,
                torch.ones_like(current), denominator=count)
            loss.backward()
    torch.nn.utils.clip_grad_norm_(backend.planner_parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    return True


def save(backend, directory, name, metadata):
    with tempfile.TemporaryDirectory(dir=directory) as tmp:
        checkpoint = Path(tmp) / "checkpoint"
        backend.model.save_pretrained(checkpoint, selected_adapters=["planner"])
        record = {**metadata, "parent": backend.parent, "fixed_skill": backend.fixed_skill,
                  "planner_sha256": sha(checkpoint / "planner/adapter_model.safetensors")}
        (checkpoint / "checkpoint.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        checkpoint.rename(directory / name)


def train(public, refs, backend, output, group_size, batch_sources):
    import torch
    optimizer = torch.optim.AdamW(backend.planner_parameters(), lr=5e-6, weight_decay=0.)
    groups = []
    cases = public["inputs"]["cases"]
    updates = 0
    with (output / "episodes.jsonl").open("x", encoding="utf-8") as log:
        for index, case in enumerate(cases, 1):
            group = []
            for k in range(group_size):
                result, actions = run_episode(public, case, backend, sample=True)
                if not actions:
                    raise RuntimeError("Missing sampled Planner actions")
                reward = int(result["valid"] and result["prediction"] == refs[case["source_id"]])
                group.append((reward, actions))
                row = {**result, "source_id": case["source_id"], "sample": k,
                       "policy_version": updates, "reward": reward}
                log.write(json.dumps(row, ensure_ascii=False) + "\n")
                log.flush()
            groups.append(group)
            if len(groups) == batch_sources or index == len(cases):
                if update(backend, optimizer, groups):
                    updates += 1
                    save(backend, output, f"step-{updates:03d}", {"updates": updates, "sources_seen": index})
                groups.clear()
    return {"sources": len(cases), "episodes": len(cases) * group_size, "updates": updates}
