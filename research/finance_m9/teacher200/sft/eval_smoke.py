"""Record the student M9 predictions, native traces and model calls."""
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import patch

STUDY = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("FINANCE_SFT_EVAL_RUN", STUDY/"work"/"sft-eval")).resolve()
M9 = STUDY.parent
sys.path.insert(0,str(M9))
sys.path.insert(0,str(STUDY/"rl"))
from gpu_resources import idle

BASE = Path(os.environ.get("FINANCE_BASE_MODEL", "Qwen3.5-9B")).resolve()
SKILL = Path(os.environ.get("FINANCE_SKILL_ADAPTER", STUDY/"weights"/"skill")).resolve()
PLANNERS = {"student": Path(os.environ.get("FINANCE_SFT_ADAPTER", STUDY/"weights"/"sft")).resolve()}


def write(path, obj):
    tmp = path.with_suffix(".partial")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2))
    tmp.replace(path)


def evaluate(arm, *, source_ids=None, output=None):
    import episode
    from model import load_backend
    from native_assets import prepare
    from transformers import set_seed
    args = SimpleNamespace(model=BASE, planner=PLANNERS[arm], skill=SKILL)
    set_seed(2026110509)
    public = episode.cohort(M9 / "val422/val422-inputs.json", "VAL")
    cases = {c["source_id"]: c for c in public["inputs"]["cases"]}
    if len(cases) != 422:
        raise ValueError("Need all VAL sources for full table lookups")
    with tempfile.TemporaryDirectory() as tmp:
        episode.install_native(prepare(Path(tmp) / "deps", public["scopes"]))
        backend = load_backend(args)
        original_call = backend.call
        original_env = episode.environment
        for sid in (cases if source_ids is None else source_ids):
            folder = (ROOT / "runs" if output is None else output) / arm / sid
            folder.mkdir(exist_ok=True, parents=True)
            if (folder / "result.json").is_file():
                continue
            calls = []
            def observe_call(stage, messages, *, sample):
                raw = json.dumps(messages, ensure_ascii=False, separators=(",", ":")).encode()
                event = dict(stage=stage, input_sha256=hashlib.sha256(raw).hexdigest(), start=time.time())
                calls.append(event)
                write(folder / "calls.json", calls)
                reply, action = original_call(stage, messages, sample=sample)
                event.update(end=time.time(), response=reply)
                write(folder / "calls.json", calls)
                return reply, action

            def record_env(*args):
                env = original_env(*args)
                rollout = env.run
                def run(*inner):
                    trace, audit = rollout(*inner)
                    write(folder / "trace.json", trace)
                    return trace, audit
                env.run = run
                return env

            with patch.object(episode, "environment", record_env), patch.object(backend, "call", observe_call):
                try:
                    result, _ = episode.run_episode(public, cases[sid], backend)
                except Exception as err:
                    import traceback
                    write(folder / "error.json", {"exception": repr(err), "traceback": traceback.format_exc()})
                    continue
                write(folder / "result.json", dict(source_id=sid, **result, planner=backend.parent, skill=backend.fixed_skill))
