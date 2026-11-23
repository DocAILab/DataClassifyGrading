"""Public inputs and shared native M9 rollout for evaluation and training."""
from copy import deepcopy
import os
from pathlib import Path
import sys
import time

from native_projection import build_public
from native_assets import ROOT, read
from resident_input_contract import clean_metadata
from retrieval_input_contract import disclose_descriptions


def cohort(path, split):
    data = read(path)
    if set(data) != {"split", "rows"} or data["split"] != split or not data["rows"]:
        raise ValueError(f"Expected a nonempty {split} metadata cohort")
    rows = []
    seen = set()
    for row in data["rows"]:
        if set(row) != {"source_id", "metadata"}:
            raise ValueError("Only source_id and observed metadata may enter the environment")
        sid = row["source_id"]
        if not isinstance(sid, str) or not sid or sid in seen:
            raise ValueError("A unique source_id is required")
        metadata = clean_metadata(row["metadata"])
        if set(row["metadata"]) - set(metadata) or not metadata.get("field_name"):
            raise ValueError("Use only observed metadata and supply field_name")
        seen.add(sid)
        rows.append({"source_id": sid, "metadata": metadata})
    template = {"inputs": {"study": read(ROOT / "assets/study.json")}}
    return build_public(template, rows)


def install_native(dependencies):
    # Historical modules use unqualified imports. Load them in an isolated CLI process.
    os.environ["FINANCE_DEPENDENCY_HOME"] = str(Path(dependencies).resolve())
    sys.path[:0] = [str(ROOT / "full15/native"), str(ROOT / "full15")]


def environment(public, case):
    from native_env import Environment
    from unresolved_guard_01 import finish

    inputs = {"inputs": {**public["inputs"], "cases": [deepcopy(case)]},
              "scopes": public["scopes"]}
    env = Environment(inputs)
    namespace = env.rollout.__globals__
    previous = namespace["finish"]
    namespace["finish"] = lambda tr, ctrl, record: finish(tr, ctrl, record, previous)
    return env


def run_episode(public, case, backend, *, sample=False):
    from prompt200 import format_messages

    env = environment(public, case)
    actions = []
    cost = 0
    deadline = time.monotonic() + 1800
    errors = []

    def generate(messages):
        nonlocal cost
        first = messages[0]["content"].splitlines()[0]
        prefix = "Execute only the selected Skill "
        stage = first[len(prefix):].removesuffix(".") if first.startswith(prefix) else "Planner"
        cost += 3 if stage.startswith("SEARCH_STANDARD_") else 1
        try:
            if cost > 64 or time.monotonic() > deadline:
                raise RuntimeError("M9 episode budget exhausted")
            prompt = disclose_descriptions(format_messages(messages), backend.registry)
            text, action = backend.call(stage, prompt, sample=sample)
            if action is not None:
                actions.append(action)
            return text
        except Exception as exc:
            # Native retries can turn a backend exception into feedback. A partial
            # group must never look like business failures with zero reward.
            errors.append(exc)
            raise

    trace, audit = env.run(env.cases[0], generate)
    if errors:
        raise RuntimeError("Incomplete native model rollout") from errors[0]
    if time.monotonic() > deadline:
        raise TimeoutError("M9 episode time limit")
    valid = trace.get("stop_reason") == "validated_terminal" and audit["native_guard_valid"]
    return {"valid": bool(valid), "prediction": trace.get("final_result"),
            "stop_reason": trace.get("stop_reason"), "calls": cost}, actions
