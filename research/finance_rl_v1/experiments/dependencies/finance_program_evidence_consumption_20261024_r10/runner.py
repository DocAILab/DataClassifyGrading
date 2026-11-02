"""Targeted8 then separately authorized same-freeze full80, reusing8 slots exactly."""
import json,os,sys,time
from common import ROOT,PARENT,put,verify,sha
import interface
from policy import targeted,pending
from backend import EvaluationBackend,ResourceFailure
from eval_runtime import prepare,ContextBackend,CPUReplay,raw_bytes,TechnicalFailure,BudgetExhausted

def main(phase):
    assert phase in ('targeted','full')
    verify();admission=json.loads((ROOT/'admission.json').read_bytes())
    slots=json.loads((PARENT/'selection.json').read_bytes())['slots'];targets=targeted(slots)
    run=ROOT/'runs';rows=[]
    if phase=='targeted':run.mkdir(exist_ok=False)
    else:
        approval=json.loads((ROOT/'reviews/semantic-gate-01.json').read_bytes())
        assert approval['passed'] is True and approval['freeze_sha256']==sha(ROOT/'freeze.json')
        assert approval['targeted_completion_sha256']==sha(run/'targeted-completion.json')
        for item in approval['targeted_files']:
            path=ROOT/item['path'];assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256']
        prior=json.loads((run/'targeted-completion.json').read_bytes())
        assert len(prior['rows'])==8 and all(r['native_replay_exact'] for r in prior['rows'])
        rows=prior['rows']
    put(run/(phase+'-started.json'),dict(pid=os.getpid(),time_unix=time.time(),admission=admission,reused_slots=[r['slot_id'] for r in rows]))
    def fixed(expected):
        by={r['slot_id']:r for r in rows}
        return [by.get(s['slot_id'],dict(s,legal_completion=False,stop_reason='not_started')) for s in expected]
    try:
        study,cases,rollout=prepare();by={c['source_id']:c for c in cases}
        backend=EvaluationBackend(admission);backend._ensure_loaded();put(run/(phase+'-model-identity.json'),backend.identity)
        def one(slot):
            if time.time()>=admission['gpu_deadline_unix']-20:raise BudgetExhausted('fixed deadline exhausted')
            case=by[slot['source_id']];folder=run/slot['slot_id'];folder.mkdir(exist_ok=False)
            adapter=interface.RecoveryAdapter(backend)
            bridge=ContextBackend(case,adapter,slot['arm'],folder,admission);row=dict(slot,legal_completion=False)
            try:
                trace=rollout(case['item'],study,bridge,case['snapshot']);put(folder/'trace.json',trace)
                replay=CPUReplay(bridge.records);again=rollout(case['item'],study,replay,case['snapshot'])
                exact=raw_bytes(trace)==raw_bytes(again) and replay.position==len(bridge.records)
                if not exact or not bridge.target_exact:raise TechnicalFailure('native exact replay/target mismatch')
                row.update(legal_completion=bool(trace['terminated']),native_replay_exact=exact,actual_target_exact=True,
                    prediction=trace['final_result'],stop_reason=trace['stop_reason'],executed_steps=trace['new_steps'],planner_attempts=len(trace['planner_attempts']))
            except (ResourceFailure,TechnicalFailure,BudgetExhausted,OSError) as exc:
                row.update(stop_reason='technical_failure',error_type=type(exc).__name__,error=str(exc));raise
            except Exception as exc:row.update(stop_reason='slot_error',error_type=type(exc).__name__,error=str(exc))
            finally:
                row['model_calls']=len(bridge.records);put(folder/'result.json',row);rows.append(row)
        selected=targets if phase=='targeted' else pending(slots,rows)
        for slot in selected:one(slot)
        if phase=='targeted':
            put(run/'targeted-completion.json',dict(stage='targeted_complete_pending_semantic_gate',rows=fixed(targets),denominator=8,
                model_calls=sum(r['model_calls'] for r in rows),time_unix=time.time()))
        else:
            put(run/'completion.json',dict(stage='full_complete',source_denominator=40,slot_denominator=80,reused_targeted_slots=8,
                rows=fixed(slots),model_calls=sum(r['model_calls'] for r in rows),time_unix=time.time()))
        return 0
    except BaseException as exc:
        put(run/(phase+'-failure.json'),dict(error_type=type(exc).__name__,error=str(exc),rows=fixed(targets if phase=='targeted' else slots),time_unix=time.time()));raise
if __name__=='__main__':raise SystemExit(main(sys.argv[1]))
