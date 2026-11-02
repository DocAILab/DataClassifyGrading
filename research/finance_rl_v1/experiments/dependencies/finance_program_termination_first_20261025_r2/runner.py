"""Generate failed42 first, then the remaining58, without any base comparison."""
from collections import Counter
import json
import os
import time
from support import ROOT, N100, load, native, put, verify


def main():
    verify()
    old = native()
    runtime = load('termination_first_live_runtime',ROOT/'runtime.py')
    study,cases,rollout = runtime.prepare()
    from live_backend import EvaluationBackend,ResourceFailure
    admission = json.loads((ROOT/'admission.json').read_bytes())
    slots = json.loads((ROOT/'selection.json').read_bytes())['slots']
    assert len(slots) == 100 and all(s['arm']=='N' for s in slots)
    by = {c['source_id']:c for c in cases}
    interface = load('termination_first_original_R11_interface',N100/'interface.py')
    run = ROOT/'runs'
    run.mkdir(exist_ok=False)
    rows = []
    put(run/'started.json',dict(pid=os.getpid(),time_unix=time.time(),slots=100,
                               failed_first=42,reused_results=0,admission=admission))
    try:
        backend = EvaluationBackend(admission)
        backend._ensure_loaded()
        put(run/'model-identity.json',backend.identity)
        for slot in slots:
            if time.time() >= admission['gpu_deadline_unix']-20:
                raise old.BudgetExhausted('new evaluation absolute deadline exhausted')
            folder = run/slot['slot_id']
            folder.mkdir(exist_ok=False)
            case = by[slot['source_id']]
            bridge = old.ContextBackend(case,interface.RecoveryAdapter(backend),'N',folder,admission)
            result = dict(slot,algorithm_completed=False,legal_completion=False,prediction=None)
            try:
                trace = rollout(case['item'],study,bridge,case['snapshot'])
                put(folder/'trace.json',trace)
                replay = old.CPUReplay(bridge.records)
                again = rollout(case['item'],study,replay,case['snapshot'])
                exact = old.raw_bytes(trace) == old.raw_bytes(again) and replay.position == len(bridge.records)
                if not exact or not bridge.target_exact:
                    raise old.TechnicalFailure('exact native replay or initial target differs')
                terminal = trace.get('termination',{})
                prediction = trace['final_result']
                result.update(algorithm_completed=bool(trace['terminated']),
                              legal_completion=bool(trace['terminated'] and prediction is not None),
                              stop_reason=trace['stop_reason'],prediction=prediction,
                              termination=terminal,native_replay_exact=True,actual_target_exact=True,
                              native_guard_valid=(trace['stop_reason']=='validated_terminal' or terminal.get('native_guard_valid',False)),
                              executed_steps=trace['new_steps'],planner_attempts=len(trace['planner_attempts']))
            finally:
                result['model_calls'] = len(bridge.records)
                put(folder/'result.json',result)
                rows.append(result)
            if len(rows) == 42:
                predictions = sum(r['legal_completion'] and r['native_guard_valid'] for r in rows)
                gate = dict(denominator=42,algorithm_completed=sum(r['algorithm_completed'] for r in rows),
                            guard_valid_predictions=predictions,required_predictions=40,
                            all_exact=all(r['native_replay_exact'] for r in rows),
                            passed=(all(r['algorithm_completed'] and r['native_replay_exact'] for r in rows) and predictions>=40),
                            rows=rows.copy(),time_unix=time.time())
                put(run/'failed42-gate.json',gate)
                print(json.dumps({k:v for k,v in gate.items() if k!='rows'}),flush=True)
                if not gate['passed']:
                    put(run/'gate-stopped.json',dict(reason='failed42_gate_not_met',denominator=100,
                                                    not_started=58,time_unix=time.time()))
                    return 2
        summary = dict(kind='new_SFT_model_generation',source_denominator=100,slot_denominator=100,
                       N_only=True,reused_results=0,rows=rows,model_calls=sum(r['model_calls'] for r in rows),
                       stop_reasons=dict(Counter(r['stop_reason'] for r in rows)),time_unix=time.time())
        verify()
        put(run/'completion.json',summary)
        print(json.dumps({k:v for k,v in summary.items() if k!='rows'}),flush=True)
        return 0
    except BaseException as exc:
        put(run/'failure.json',dict(error_type=type(exc).__name__,error=str(exc),rows=rows,
                                   source_denominator=100,not_started=100-len(rows),time_unix=time.time()))
        raise


if __name__ == '__main__':
    raise SystemExit(main())
