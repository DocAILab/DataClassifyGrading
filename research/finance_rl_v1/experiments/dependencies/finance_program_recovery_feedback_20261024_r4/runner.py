"""Fixed20N gate then automatic full40 F/N, no response repair or retries."""
import json,os,sys,time
from common import ROOT,PARENT,put,verify,sha
import interface
from policy import gate_slots,gate_pass,pending_slots
sys.path.append(str(PARENT))
from backend import EvaluationBackend,ResourceFailure
from eval_runtime import prepare,ContextBackend,CPUReplay,raw_bytes,TechnicalFailure,BudgetExhausted
def main():
    verify();admission=json.loads((ROOT/'admission.json').read_bytes())
    slots=json.loads((PARENT/'selection.json').read_bytes())['slots'];gate=gate_slots(slots)
    run=ROOT/'runs';run.mkdir(exist_ok=False);rows=[]
    put(run/'started.json',dict(pid=os.getpid(),time_unix=time.time(),admission=admission))
    def fixed(expected):
        by={r['slot_id']:r for r in rows}
        return [by.get(s['slot_id'],dict(s,legal_completion=False,stop_reason='not_started')) for s in expected]
    try:
        study,cases,rollout=prepare();by={c['source_id']:c for c in cases}
        backend=EvaluationBackend(admission);backend._ensure_loaded();put(run/'model-identity.json',backend.identity)
        def one(slot):
            if time.time()>=admission['gpu_deadline_unix']-20:raise BudgetExhausted('campaign deadline exhausted')
            case=by[slot['source_id']];folder=run/slot['slot_id'];folder.mkdir(exist_ok=False)
            adapter=interface.RecoveryAdapter(backend)
            bridge=ContextBackend(case,adapter,slot['arm'],folder,admission);row=dict(slot,legal_completion=False)
            try:
                trace=rollout(case['item'],study,bridge,case['snapshot']);put(folder/'trace.json',trace)
                replay=CPUReplay(bridge.records);again=rollout(case['item'],study,replay,case['snapshot'])
                exact=raw_bytes(trace)==raw_bytes(again) and replay.position==len(bridge.records)
                if not exact or not bridge.target_exact:raise TechnicalFailure('native replay/initial target mismatch')
                row.update(legal_completion=bool(trace['terminated']),native_replay_exact=exact,actual_target_exact=True,
                    prediction=trace['final_result'],stop_reason=trace['stop_reason'],executed_steps=trace['new_steps'],planner_attempts=len(trace['planner_attempts']))
            except (ResourceFailure,TechnicalFailure,BudgetExhausted,OSError) as exc:
                row.update(stop_reason='technical_failure',error_type=type(exc).__name__,error=str(exc));raise
            except Exception as exc:row.update(stop_reason='slot_error',error_type=type(exc).__name__,error=str(exc))
            finally:
                row['model_calls']=len(bridge.records);put(folder/'result.json',row);rows.append(row)
        for slot in gate:one(slot)
        accepted=gate_pass(rows,gate)
        put(run/'gate-result.json',dict(rows=fixed(gate),denominator=20,legal=sum(r['legal_completion'] for r in rows),passed=accepted,time_unix=time.time()))
        if not accepted:
            put(run/'completion.json',dict(stage='gate_not_met',gate_passed=False,rows=fixed(gate),denominator=20,model_calls=sum(r['model_calls'] for r in rows),time_unix=time.time()))
            return 0
        put(run/'full-started.json',dict(time_unix=time.time(),reused_gate_slot_ids=[r['slot_id'] for r in rows],additional_slots=60))
        for slot in pending_slots(slots,rows):one(slot)
        put(run/'completion.json',dict(stage='full_complete',gate_passed=True,source_denominator=40,slot_denominator=80,
            rows=fixed(slots),model_calls=sum(r['model_calls'] for r in rows),time_unix=time.time()))
        return 0
    except BaseException as exc:
        put(run/'failure.json',dict(error_type=type(exc).__name__,error=str(exc),rows=fixed(slots),gate_denominator=20,full_denominator=80,time_unix=time.time()));raise
if __name__=='__main__':raise SystemExit(main())
