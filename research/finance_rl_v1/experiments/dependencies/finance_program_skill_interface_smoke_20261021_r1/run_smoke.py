"""One fixed B80 prompt-only diagnostic: no training or teacher calls."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
E2 = ROOT.parent / 'finance_program_student_b80_eval_20261021_r2'
sys.path.insert(0, str(E2))
from eval_backend import ModelBackend
from paired_eval import ContextBackend, CPUReplay, load_engine, load_inputs, preflight_case
from prepare_runtime import exclusive_json, raw_bytes, require
from training_contract import file_sha256, check_deadline
from contracts import rewrite, validate_output


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


class Adapter:
    def __init__(self, backend, condition):
        self.backend, self.condition = backend, condition
        self.last_generation_metadata = {}
        self.last_call_evidence = {}

    def generate(self, messages, arm, stage, destination):
        actual = rewrite(messages, self.condition)
        try:
            return self.backend.generate(actual, 'B', stage, destination)
        finally:
            self.last_generation_metadata = deepcopy(self.backend.last_generation_metadata)
            self.last_call_evidence = deepcopy(self.backend.last_call_evidence)


def main():
    run = ROOT / 'runs'
    run.mkdir(exist_ok=False)
    admission = json.loads((ROOT / 'admission.json').read_bytes())
    exclusive_json(run / 'started.json', dict(pid=__import__('os').getpid(), time_unix=time.time(), admission=admission))
    inputs = json.loads((ROOT / 'inputs.json').read_bytes())
    study, _ = load_inputs(E2)
    engine = load_engine(E2)
    cases = study['cases'][:4]
    require([c['source_id'] for c in cases] == inputs['case_source_ids'], 'first four initial sources changed')
    require(file_sha256(E2 / 'runtime-inputs/inputs.json') == inputs['parent_inputs_sha256'], 'parent inputs changed')
    exclusive_json(run / 'native-preflight.json', [preflight_case(engine, study['study'], c) for c in cases])
    backend = ModelBackend(E2, admission)
    backend._ensure_loaded()
    backend._select('B')
    exclusive_json(run / 'model-identity.json', backend.identity)
    vendor = E2 / 'vendor'
    g = load('smoke_grounding', vendor / 'finance_program_student_grpo_20261017_r3/runtime/vendor/grounded_semantics.py')
    b = load('smoke_boundary', vendor / 'finance_program_teacher_evidence_20261020_r1/boundary.py')
    fixed_results, native_results = [], []
    for index, case in enumerate(inputs['fixed_calls']):
        for condition in (('original', 'unified') if index % 2 == 0 else ('unified', 'original')):
            check_deadline(admission)
            folder = run / f'fixed-{index+1:02d}-{condition}'
            folder.mkdir()
            actual = rewrite(case['messages'], condition)
            record = dict(index=index+1, stage=case['stage'], source_id=case['source_id'], condition=condition,
                          original_path=case['path'], original_file_sha256=case['file_sha256'])
            try:
                raw = backend.generate(actual, 'B', case['stage'], folder)
                record.update(raw=raw, validation=validate_output(case['stage'], actual, raw, g, b))
            except Exception as exc:
                record.update(error_type=type(exc).__name__, error=str(exc), validation=dict(protocol_valid=False, accepted_success=False))
            exclusive_json(folder / 'result.json', record)
            fixed_results.append(record)
    for index, case in enumerate(cases):
        for condition in (('original', 'unified') if index % 2 == 0 else ('unified', 'original')):
            check_deadline(admission)
            folder = run / f'native-{index+1:02d}-{condition}'
            folder.mkdir()
            bridge = ContextBackend(case, Adapter(backend, condition), 'B', folder, admission)
            result = dict(index=index+1, source_id=case['source_id'], condition=condition, legal_completion=False)
            try:
                trace = engine.rollout(case['item'], study['study'], bridge, case['snapshot'])
                exclusive_json(folder / 'trace.json', trace)
                replay = CPUReplay(bridge.records)
                recovered = engine.rollout(case['item'], study['study'], replay, case['snapshot'])
                exact = raw_bytes(trace) == raw_bytes(recovered) and replay.position == len(replay.records)
                require(exact, 'original native replay differs')
                result.update(legal_completion=trace['terminated'], prediction=trace['final_result'],
                              stop_reason=trace['stop_reason'], native_replay_exact=exact)
            except Exception as exc:
                result.update(error_type=type(exc).__name__, error=str(exc))
            result.update(model_calls=len(bridge.records), actual_target_exact=bridge.target_exact)
            exclusive_json(folder / 'result.json', result)
            native_results.append(result)
    exclusive_json(run / 'completion.json', dict(time_unix=time.time(), training_updates=0, teacher_api_calls=0,
        fixed_denominator=2*len(inputs['fixed_calls']), native_denominator=8,
        fixed_results=fixed_results, native_results=native_results, final_delivery_complete=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
