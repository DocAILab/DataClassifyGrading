"""Full422 N-only evaluation, once per fixed source, with exact native replay."""
import json,os,time
from support import ROOT,N100,load,native,put,verify
from full_policy import summarize
from gate import assess


def main():
    verify();old=native();parent=json.loads((ROOT/'candidate-request.json').read_bytes())
    study,cases,rollout=load('full422_live_runtime',ROOT/'runtime.py').prepare()
    from live_backend import EvaluationBackend
    slots=json.loads((ROOT/'selection.json').read_bytes())['slots']
    admission=json.loads((ROOT/'admission.json').read_bytes())
    assert len(slots)==422 and all(s['arm']=='N' for s in slots)
    interface=load('full422_exact_R11_interface',N100/'interface.py')
    by={c['source_id']:c for c in cases};rows=[];run=ROOT/'runs';run.mkdir(exist_ok=False)
    put(run/'started.json',dict(pid=os.getpid(),time_unix=time.time(),denominator=422,parent=parent,admission=admission))
    try:
        backend=EvaluationBackend(admission);backend._ensure_loaded();put(run/'model-identity.json',backend.identity)
        for slot in slots:
            if time.time()>=admission['gpu_deadline_unix']-20:raise old.BudgetExhausted('full422 fixed evaluation deadline exhausted')
            case=by[slot['source_id']];folder=run/slot['slot_id'];folder.mkdir(exist_ok=False)
            bridge=old.ContextBackend(case,interface.RecoveryAdapter(backend),'N',folder,admission)
            result=dict(slot,algorithm_completed=False,legal_completion=False,prediction=None,stop_reason='technical_failure')
            try:
                trace=rollout(case['item'],study,bridge,case['snapshot']);put(folder/'trace.json',trace)
                replay=old.CPUReplay(bridge.records);again=rollout(case['item'],study,replay,case['snapshot'])
                exact=old.raw_bytes(trace)==old.raw_bytes(again) and replay.position==len(bridge.records)
                if not exact or not bridge.target_exact:raise old.TechnicalFailure('full422 native replay or initial target differs')
                terminal=trace.get('termination',{});prediction=trace['final_result']
                result.update(algorithm_completed=bool(trace['terminated']),legal_completion=bool(trace['terminated'] and prediction is not None),
                              stop_reason=trace['stop_reason'],prediction=prediction,termination=terminal,
                              native_replay_exact=True,actual_target_exact=True,
                              native_guard_valid=trace['stop_reason']=='validated_terminal' or terminal.get('native_guard_valid',False),
                              executed_steps=trace['new_steps'],planner_attempts=len(trace['planner_attempts']))
            except BaseException as exc:
                result.update(error_type=type(exc).__name__,error=str(exc));raise
            finally:
                result['model_calls']=len(bridge.records);put(folder/'result.json',result);rows.append(result)
            if len(rows)==20:
                gate=assess(rows)
                put(run/'gate20.json',dict(gate,rows=rows.copy(),time_unix=time.time()))
                print(json.dumps(gate),flush=True)
                # Report the historical first-20 diagnostic without changing the fixed cohort.
        verify()
        put(run/'completion.json',dict(kind='GRPO_only_fullVal422_generation',N_only=True,source_denominator=422,reused_results=0,
                                       rows=rows,model_calls=sum(r['model_calls'] for r in rows),summary=summarize(rows),time_unix=time.time()))
        return 0
    except BaseException as exc:
        present={r['source_id'] for r in rows}
        missing=[dict(s,algorithm_completed=False,legal_completion=False,prediction=None,stop_reason='not_started') for s in slots if s['source_id'] not in present]
        put(run/'failure.json',dict(error_type=type(exc).__name__,error=str(exc),rows=rows+missing,summary=summarize(rows),time_unix=time.time()))
        raise


if __name__=='__main__':raise SystemExit(main())
