"""One fixed20 end-to-end run, latest prompt and 48K, no training or labels."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
path = ROOT.parent / 'finance_program_skill_recovery20_20261021_r1/run_smoke.py'
spec = importlib.util.spec_from_file_location('pinned_fixed20_preparation', path)
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
previous.ROOT = ROOT
prepare, preflight = previous.prepare, previous.preflight
from paired_eval import ContextBackend, CPUReplay, TechnicalFailure, BudgetExhausted
from prepare_runtime import exclusive_json, raw_bytes, require
from training_contract import check_deadline
from backend import backend_class
from interface import rewrite
from runtime import limits_remaining

class ResourceFailure(BaseException):
    pass

class Adapter(previous.Adapter):
    def generate(self, messages, arm, stage, destination):
        try:
            return self.backend.generate(rewrite(messages), 'B', stage, destination)
        except Exception as exc:
            if 'out of memory' in str(exc).lower() or type(exc).__name__ == 'OutOfMemoryError':
                raise ResourceFailure(str(exc)) from exc
            raise
        finally:
            from copy import deepcopy
            self.last_generation_metadata = deepcopy(self.backend.last_generation_metadata)
            self.last_call_evidence = deepcopy(self.backend.last_call_evidence)

def main():
    from launch_smoke import verify
    verify()
    run = ROOT/'runs'
    run.mkdir(exist_ok=False)
    admission = json.loads((ROOT/'admission.json').read_bytes())
    exclusive_json(run/'started.json', dict(pid=os.getpid(), time_unix=time.time(), admission=admission))
    exclusive_json(run/'native-preflight.json', preflight())
    study, cases, rollout = prepare()
    backend = backend_class(previous.E2)(previous.E2, admission)
    rows = []
    try:
        backend._ensure_loaded()
        backend._select('B')
        exclusive_json(run/'model-identity.json', backend.identity)
        for index, case in enumerate(cases, 1):
            check_deadline(admission)
            folder = run/f'native-{index:02d}-unified'
            folder.mkdir()
            row = dict(index=index, source_id=case['source_id'], condition='unified', kind='initial', legal_completion=False)
            bridge = ContextBackend(case, Adapter(backend), 'B', folder, admission)
            try:
                trace = rollout(case['item'], study, bridge, case['snapshot'])
                exclusive_json(folder/'trace.json', trace)
                replay = CPUReplay(bridge.records)
                recovered = rollout(case['item'], study, replay, case['snapshot'])
                exact = raw_bytes(trace) == raw_bytes(recovered) and replay.position == len(replay.records)
                require(exact, 'native replay differs')
                row.update(legal_completion=trace['terminated'], prediction=trace['final_result'], stop_reason=trace['stop_reason'],
                           native_replay_exact=exact, planner_attempts=len(trace['planner_attempts']), executed_steps=trace['new_steps'])
            except (ResourceFailure, BudgetExhausted, TechnicalFailure) as exc:
                row.update(error_type=type(exc).__name__, error=str(exc), model_calls=len(bridge.records), actual_target_exact=bridge.target_exact)
                exclusive_json(folder/'result.json', row)
                rows.append(row)
                raise
            except Exception as exc:
                row.update(error_type=type(exc).__name__, error=str(exc))
                if isinstance(exc, OSError):
                    raise
            row.update(model_calls=len(bridge.records), actual_target_exact=bridge.target_exact)
            exclusive_json(folder/'result.json', row)
            rows.append(row)
        exclusive_json(run/'completion.json', dict(native_denominator=20, fixed_denominator=0, native_results=rows,
            training_updates=0, teacher_api_calls=0, time_unix=time.time(), budget=limits_remaining(1, 0)))
        return 0
    except BaseException as exc:
        exclusive_json(run/'failure.json', dict(error_type=type(exc).__name__, error=str(exc), time_unix=time.time(),
                                              fixed_denominator=20, closed_rows=rows, unattempted=20-len(rows)))
        raise

if __name__ == '__main__':
    if sys.argv[1:] == ['--preflight']:
        print(json.dumps(preflight()))
    else:
        raise SystemExit(main())
