"""Fixed isolated40 F/N paired evaluation, automatically after successful SFT."""
import json
import os
import time
from contract import ROOT,put
from deps import identity
from eval_runtime import prepare,preflight,interface,ContextBackend,CPUReplay,raw_bytes,TechnicalFailure,BudgetExhausted
from model_backend import ModelBackend

class ResourceFailure(BaseException):pass
class Adapter:
    def __init__(self,backend):self.backend=backend;self.last_generation_metadata={};self.last_call_evidence={}
    def generate(self,messages,arm,stage,destination):
        try:return self.backend.generate(interface.rewrite(messages),arm,stage,destination)
        except Exception as exc:
            if 'out of memory' in str(exc).lower() or type(exc).__name__=='OutOfMemoryError':raise ResourceFailure(str(exc)) from exc
            raise
        finally:
            self.last_generation_metadata=self.backend.last_generation_metadata.copy();self.last_call_evidence=self.backend.last_call_evidence.copy()
def run():
    from supervisor import verify
    verify();run=ROOT/'runs/evaluation';run.mkdir(exist_ok=False)
    admission=json.loads((ROOT/'admission.json').read_bytes());slots=json.loads((ROOT/'selection.json').read_bytes())['slots'];rows=[]
    put(run/'started.json',dict(pid=os.getpid(),admission=admission,time_unix=time.time()))
    put(run/'preflight.json',preflight());study,cases,rollout=prepare();by_id={c['source_id']:c for c in cases}
    def fixed():
        by={r['slot_id']:r for r in rows}
        return [by.get(s['slot_id'],dict(s,legal_completion=False,stop_reason='not_started')) for s in slots]
    try:
        backend=ModelBackend(admission);backend._ensure_loaded();put(run/'model-identity.json',backend.identity);adapter=Adapter(backend)
        for slot in slots:
            identity.check_deadline(admission);folder=run/slot['slot_id'];folder.mkdir(exist_ok=False)
            case=by_id[slot['source_id']];bridge=ContextBackend(case,adapter,slot['arm'],folder,admission);row=dict(slot,legal_completion=False)
            try:
                trace=rollout(case['item'],study,bridge,case['snapshot']);put(folder/'trace.json',trace)
                replay=CPUReplay(bridge.records);again=rollout(case['item'],study,replay,case['snapshot'])
                exact=raw_bytes(trace)==raw_bytes(again) and replay.position==len(bridge.records)
                if not exact or not bridge.target_exact:raise TechnicalFailure('native replay/initial message differs')
                row.update(legal_completion=bool(trace['terminated']),native_replay_exact=exact,actual_target_exact=bridge.target_exact,
                    prediction=trace['final_result'],stop_reason=trace['stop_reason'],executed_steps=trace['new_steps'],planner_attempts=len(trace['planner_attempts']))
            except (ResourceFailure,TechnicalFailure,BudgetExhausted,OSError) as exc:
                row.update(stop_reason='campaign_technical_failure',error_type=type(exc).__name__,error=str(exc));raise
            except Exception as exc:row.update(stop_reason='slot_error',error_type=type(exc).__name__,error=str(exc))
            finally:row.update(model_calls=len(bridge.records));put(folder/'result.json',row);rows.append(row)
        put(run/'completion.json',dict(source_denominator=40,slot_denominator=80,rows=fixed(),teacher_api_calls=0,time_unix=time.time()))
        return 0
    except BaseException as exc:
        put(run/'failure.json',dict(error_type=type(exc).__name__,error=str(exc),rows=fixed(),closed=len(rows),time_unix=time.time()));raise
if __name__=='__main__':raise SystemExit(run())
