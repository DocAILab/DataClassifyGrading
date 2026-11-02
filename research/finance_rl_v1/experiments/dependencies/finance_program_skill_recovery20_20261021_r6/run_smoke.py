"""Exactly one new Skill response, no Planner rollout or reference labels."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import time

ROOT = Path(__file__).resolve().parent
E2 = ROOT.parent / 'finance_program_student_b80_eval_20261021_r2'
sys.path.insert(0, str(E2))
sys.path.insert(0, str(ROOT))
from prepare_runtime import exclusive_json
from paired_eval import load_engine
from guarded_adapter import GuardedAdapter
from interface import rewrite

def main():
    from launch_smoke import verify
    verify()
    run = ROOT / 'runs'
    run.mkdir(exist_ok=False)
    admission = json.loads((ROOT / 'admission.json').read_bytes())
    exclusive_json(run / 'started.json', dict(pid=os.getpid(), time_unix=time.time(), admission=admission))
    inputs = json.loads((ROOT / 'input.json').read_bytes())
    load_engine(E2)
    from agent.orchestration.model_skill_executor import parse_skill_observation
    skill = SimpleNamespace(name='SYNTHESIZE_METADATA_SEMANTICS', produces=('metadata_semantics',))
    try:
        parse_skill_observation(inputs['old_raw'], skill)
    except Exception as exc:
        old_error = dict(error_type=type(exc).__name__, error=str(exc))
    else:
        raise AssertionError('original malformed output unexpectedly accepted')
    exclusive_json(run / 'original-native-rejection.json', old_error)
    path = ROOT.parent / 'finance_program_skill_recovery20_20261021_r5/backend.py'
    spec = importlib.util.spec_from_file_location('unchanged_loaded_backend', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    backend = module.backend_class(E2)(E2, admission)
    try:
        backend._ensure_loaded()
        backend._select('B')
        exclusive_json(run / 'model-identity.json', backend.identity)
        call = run / 'call-001'
        call.mkdir()
        adapter = GuardedAdapter(backend, rewrite)
        raw = adapter.generate(inputs['messages'], 'B', skill.name, call)
        result = dict(denominator=1, actual_new_model_calls=1, original_native_rejection=old_error,
            original_raw_sha256=hashlib.sha256(inputs['old_raw'].encode('utf-8')).hexdigest(),
            new_raw_sha256=hashlib.sha256(raw.encode('utf-8')).hexdigest(), native_valid=False)
        try:
            observation = parse_skill_observation(raw, skill)
            assert isinstance(observation.data['metadata_semantics'], str) and observation.data['metadata_semantics'].strip()
            result.update(native_valid=True, summary=observation.summary, data=observation.data)
        except Exception as exc:
            result.update(error_type=type(exc).__name__, error=str(exc))
        exclusive_json(run / 'result.json', result)
        exclusive_json(run / 'completion.json', dict(result=result, time_unix=time.time(), training_updates=0, teacher_api_calls=0,
            meaning='one targeted formatting check; not new fixed20 accuracy or end-to-end recovery proof'))
        return 0
    except BaseException as exc:
        exclusive_json(run / 'failure.json', dict(error_type=type(exc).__name__, error=str(exc), time_unix=time.time()))
        raise

if __name__ == '__main__':
    raise SystemExit(main())
