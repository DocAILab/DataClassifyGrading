"""Fixed paired slots through the original H engine; no labels or teacher access."""
import argparse
from copy import deepcopy
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
import time

from training_contract import ProtocolError, canonical_json_bytes, check_deadline, file_sha256, sha256_bytes, validate_admission
from prepare_runtime import exclusive_json, raw_bytes, require

_ENGINES = {}


class TargetReached(BaseException):
    """CPU boundary interception bypasses native exception-to-feedback handling."""


class TechnicalFailure(BaseException):
    pass


class BudgetExhausted(BaseException):
    pass


def load_engine(root):
    root = Path(root).resolve(strict=True)
    if root in _ENGINES:
        return _ENGINES[root]
    vendor = root / 'vendor'
    receipt_path = root / 'reviews/runtime-copy-receipt-01.json'
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_bytes())
        assets = {entry['target']: entry['sha256'] for entry in receipt['files']}
    else:
        assets = json.loads((root / 'runtime-inputs/manifest.json').read_bytes())['vendor_files_sha256']
    require(len(assets) == 121, 'exact native asset manifest required')
    for relative, digest in assets.items():
        path = root / relative
        require(relative.startswith('vendor/') and not path.is_symlink() and
                path.resolve(strict=True).is_relative_to(vendor) and file_sha256(path) == digest,
                'native vendor identity mismatch')
    # Imports below use historical module aliases. Never reuse an old project's modules.
    for name, module in tuple(sys.modules.items()):
        if name.split('.')[0] in ('agent', 'program_loop') or name.startswith(('segment_original_', 'gap_', 'wide_original_')):
            path = getattr(module, '__file__', None)
            require(path is not None and Path(path).resolve().is_relative_to(vendor),
                    'historical module alias outside the frozen vendor: ' + name)
    path = vendor / 'finance_program_teacher_evidence_20261020_r2/engine.py'
    spec = importlib.util.spec_from_file_location('b80_native_h_r2', path)
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    runtime = vendor / 'finance_program_student_grpo_20261017_r3/runtime'
    require(Path(engine.__file__).resolve() == path and
            Path(engine.old.c.controller.__defaults__[0]).resolve() == runtime,
            'native H/controller must resolve only to copied vendor')
    _ENGINES[root] = engine
    return engine


def stage_for(messages):
    match = re.match(r'Execute only the selected Skill ([A-Z_]+)\.', messages[0]['content'])
    return match.group(1) if match else 'Planner'


def message_bytes(messages):
    # Data selection hashes canonical JSON of the message envelope. Content is
    # an opaque exact string: never parse, reorder or rewrite its PromptRefs.
    require(isinstance(messages, list) and all(set(m) == {'role', 'content'} and
            isinstance(m['role'], str) and isinstance(m['content'], str) for m in messages),
            'exact role/content message envelopes required')
    return canonical_json_bytes(messages)


class CPUReplay:
    """Replay adapted record.response and original generation, never raw_content."""
    def __init__(self, records):
        self.records = records
        self.position = 0
        self.last_generation_metadata = {}

    def __call__(self, messages):
        if self.position >= len(self.records):
            raise TechnicalFailure('native replay requested an unrecorded model call')
        record = self.records[self.position]
        if message_bytes(messages) != message_bytes(record['messages']):
            raise TechnicalFailure('native replay request/stage/order differs from recorded student messages')
        self.position += 1
        self.last_generation_metadata = deepcopy(record['generation'])
        if record['error']:
            raise RuntimeError(record['error'])
        return record['response']


class ContextBackend:
    def __init__(self, case, backend, arm, directory, admission=None, probe=False):
        self.case, self.backend, self.arm, self.directory = case, backend, arm, Path(directory)
        self.admission, self.probe = admission, probe
        self.prefix = CPUReplay(case['prefix'])
        self.records = []
        self.target_exact = False
        self.last_generation_metadata = {}

    def __call__(self, messages):
        if self.prefix.position < len(self.prefix.records):
            try:
                return self.prefix(messages)
            finally:
                self.last_generation_metadata = deepcopy(self.prefix.last_generation_metadata)
        if not self.target_exact:
            if stage_for(messages) != 'Planner' or message_bytes(messages) != message_bytes(self.case['messages']):
                raise TechnicalFailure('first new request differs from locked student target')
            self.target_exact = True
            if self.probe:
                raise TargetReached()
        if self.admission is not None:
            try:
                check_deadline(self.admission)
            except ProtocolError as exc:
                raise BudgetExhausted(str(exc)) from exc
        index = len(self.records) + 1
        call = self.directory / f'call-{index:03d}'
        call.mkdir(exist_ok=False)
        started = time.time()
        record = dict(messages=deepcopy(messages), arm=self.arm, stage=stage_for(messages),
                      response=None, generation={}, error=None, started_unix=started,
                      messages_sha256=sha256_bytes(raw_bytes(messages)), backend_invoked=False, backend_returned=False)
        exclusive_json(call / 'started.json', record)
        # Metadata from an earlier call must never leak into a failed new call.
        self.backend.last_generation_metadata = {}
        self.backend.last_call_evidence = {}
        try:
            record['backend_invoked'] = True
            record['response'] = self.backend.generate(messages, self.arm, record['stage'], call)
            record['backend_returned'] = True
            return record['response']
        except BudgetExhausted:
            record['error'] = 'absolute_gpu_deadline_exhausted'
            raise
        except Exception as exc:
            record['error'] = str(exc)
            record['error_type'] = type(exc).__name__
            # The same RuntimeError is used during CPU replay, preserving native trace.
            raise RuntimeError(record['error']) from exc
        finally:
            record['generation'] = deepcopy(getattr(self.backend, 'last_generation_metadata', {}))
            self.last_generation_metadata = deepcopy(record['generation'])
            record['model_execution'] = deepcopy(getattr(self.backend, 'last_call_evidence', {}))
            record['returned_unix'] = time.time()
            record['elapsed_seconds'] = max(0., record['returned_unix'] - started)
            exclusive_json(call / 'returned.json', record)
            self.records.append(record)


def preflight_case(engine, study, case):
    bridge = ContextBackend(case, None, None, Path('.'), probe=True)
    try:
        engine.rollout(case['item'], study, bridge, case['snapshot'])
    except TargetReached:
        require(bridge.target_exact and bridge.prefix.position == len(case['prefix']), 'prefix not wholly consumed')
        return dict(source_id=case['source_id'], kind=case['kind'], first_new_student_messages_exact=True,
                    prefix_teacher_replay=bridge.prefix.position, new_student_calls=0)
    raise ProtocolError('native rollout ended before locked target')


def preflight(root):
    inputs, manifest = load_inputs(root)
    engine = load_engine(root)
    results = []
    for case in inputs['cases']:
        try:
            results.append(preflight_case(engine, inputs['study'], case))
        except TechnicalFailure as exc:
            raise ProtocolError(str(exc)) from exc
    return dict(inputs_sha256=manifest['inputs_sha256'], cases=results, new_model_calls=0)


def load_inputs(root, cpu_test=False):
    root = Path(root)
    path = root / 'runtime-inputs/inputs.json'
    manifest = json.loads((root / 'runtime-inputs/manifest.json').read_bytes())
    require(not path.is_symlink() and file_sha256(path) == manifest['inputs_sha256'], 'runtime inputs SHA mismatch')
    inputs = json.loads(path.read_bytes())
    if not cpu_test:
        from training_contract import MANIFEST_SHA256
        require(manifest['selection_manifest_sha256'] == MANIFEST_SHA256 and
                [c['source_id'] for c in inputs['cases']] == manifest['source_order'], 'fixed runtime source order mismatch')
        require([c['kind'] for c in inputs['cases']] == ['initial'] * 18 + ['recovery'] * 12,
                'fixed 18+12 sources required')
        require(set(inputs['study']) == {'registry', 'rows', 'budget'}, 'label-free study shape required')
    return inputs, manifest


def run(root, admission, backend=None):
    root = Path(root)
    test = backend is not None
    require(not test or getattr(backend, 'cpu_test_only', False) is True, 'injection is synthetic CPU-only')
    if not test:
        deadline = admission.get('gpu_deadline_unix')
        require(type(deadline) in (int, float) and math.isfinite(deadline), 'finite original absolute deadline required')
        # Validate resource vectors even if the original budget has already ended;
        # the fixed denominator must still be written as unstarted below.
        validate_admission(admission, now=min(time.time(), deadline - 1))
    inputs, manifest = load_inputs(root, test)
    engine = load_engine(getattr(backend, 'engine_root', root))
    parent = root / 'runs'
    require(not parent.is_symlink() and (not parent.exists() or parent.resolve() == parent.absolute()),
            'runs may not alias another root')
    parent.mkdir(exist_ok=True)
    directory = parent / 'evaluation'
    directory.mkdir(exist_ok=False)
    exclusive_json(directory / 'claim.json', dict(pid=os.getpid(), time_unix=time.time(), admission=admission,
                                                 runtime_manifest_sha256=file_sha256(root / 'runtime-inputs/manifest.json')))
    # Check every target before any model load/call. A bad fixed case remains a failed slot.
    prechecks = []
    for case in inputs['cases']:
        try:
            prechecks.append(preflight_case(engine, inputs['study'], case))
        except (Exception, TechnicalFailure) as exc:
            prechecks.append(dict(error_type=type(exc).__name__, error=str(exc)))
    exclusive_json(directory / 'preflight.json', prechecks)
    initialization_error = None
    if not test and time.time() < admission['gpu_deadline_unix']:
        from eval_backend import ModelBackend
        try:
            backend = ModelBackend(root, admission)
        except Exception as exc:
            initialization_error = dict(error_type=type(exc).__name__, error=str(exc))
            exclusive_json(directory / 'backend-initialization-error.json', initialization_error)
    slots, exhausted = [], False
    for source_index, case in enumerate(inputs['cases']):
        for arm in (('A', 'B') if source_index % 2 == 0 else ('B', 'A')):
            slot = dict(slot=len(slots) + 1, source_id=case['source_id'], kind=case['kind'], arm=arm,
                        inputs_sha256=manifest['inputs_sha256'], case_sha256=sha256_bytes(raw_bytes(case)),
                        prefix_teacher_replay=0, new_student_calls=0)
            target = directory / f'slot-{slot["slot"]:03d}'
            target.mkdir(exist_ok=False)
            try:
                check_deadline(admission)
            except ProtocolError:
                exhausted = True
            if exhausted:
                slot['status'] = 'unstarted_budget_exhausted'
                exclusive_json(target / 'result.json', slot)
                slots.append(slot)
                continue
            exclusive_json(target / 'started.json', dict(slot, time_unix=time.time(),
                           backend_identity=getattr(backend, 'identity', {'synthetic_cpu_only': test})))
            bridge = ContextBackend(case, backend, arm, target, admission)
            try:
                if 'error' in prechecks[source_index]:
                    raise TechnicalFailure(prechecks[source_index]['error'])
                if initialization_error is not None:
                    raise TechnicalFailure('backend initialization failed: ' + initialization_error['error'])
                trace = engine.rollout(case['item'], inputs['study'], bridge, case['snapshot'])
                if not bridge.target_exact or bridge.prefix.position != len(case['prefix']):
                    raise TechnicalFailure('native target not reached / prefix not fully consumed')
                exclusive_json(target / 'trace.json', trace)
                replay = CPUReplay(case['prefix'] + bridge.records)
                recovered = engine.rollout(case['item'], inputs['study'], replay, case['snapshot'])
                exact = raw_bytes(trace) == raw_bytes(recovered)
                slot['native_replay'] = dict(trace_exact=exact, all_calls_consumed=replay.position == len(replay.records),
                                            first_request_exact=bridge.target_exact, new_model_calls=0,
                                            meaning='verification of already generated fixed records')
                exclusive_json(target / 'native-replay.json', dict(trace=recovered, **slot['native_replay']))
                if not exact or replay.position != len(replay.records):
                    raise TechnicalFailure('native CPU replay differs from actual trace')
                slot.update(status='completed' if trace['terminated'] else 'not_terminated',
                            legal_completion=trace['terminated'], prediction=trace['final_result'],
                            stop_reason=trace['stop_reason'], total_native_steps_including_prefix=trace['new_steps'])
            except BudgetExhausted as exc:
                exhausted = True
                slot.update(status='budget_exhausted', error=str(exc), legal_completion=False)
            except (Exception, TechnicalFailure) as exc:
                slot.update(status='technical_failure', error_type=type(exc).__name__, error=str(exc), legal_completion=False)
            finally:
                slot.update(prefix_teacher_replay=bridge.prefix.position, new_student_calls=len(bridge.records),
                            conditional_teacher_prefix=case['kind'] == 'recovery',
                            new_student_input_tokens=sum((r['generation'].get('input_tokens') or 0) for r in bridge.records),
                            new_student_output_tokens=sum((r['generation'].get('output_tokens') or 0) for r in bridge.records),
                            new_student_elapsed_seconds=sum(r['elapsed_seconds'] for r in bridge.records))
                exclusive_json(target / 'result.json', slot)
                slots.append(slot)
    result = dict(schema='b80-fixed-native-paired-evaluation-v1', cpu_test_only=test,
                  denominator=2 * len(inputs['cases']), initial_denominator=2 * sum(c['kind'] == 'initial' for c in inputs['cases']),
                  recovery_denominator=2 * sum(c['kind'] == 'recovery' for c in inputs['cases']),
                  labels_read=False, gpu_deadline_unix=admission['gpu_deadline_unix'], slots=slots)
    exclusive_json(directory / 'completion.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--admission', type=Path)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if args.preflight:
        if args.admission is not None:
            parser.error('--preflight is CPU-only and takes no admission')
        print(json.dumps(preflight(args.root), ensure_ascii=False))
    else:
        if args.admission is None:
            parser.error('--admission is required')
        print(json.dumps(run(args.root, json.loads(args.admission.read_bytes())), ensure_ascii=False))
