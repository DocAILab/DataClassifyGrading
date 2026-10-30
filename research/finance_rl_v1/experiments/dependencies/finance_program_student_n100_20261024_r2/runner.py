"""N-only100 actual evaluation, immutable8 reused and92 new slots."""
import json,os,shutil,time
from common import ROOT,PREVIOUS,put,verify,sha
import interface
from run_policy import pending
from backend import EvaluationBackend,ResourceFailure
from eval_runtime import prepare,ContextBackend,CPUReplay,raw_bytes,TechnicalFailure,BudgetExhausted
def main():
 verify();admission=json.loads((ROOT/'admission.json').read_bytes());slots=json.loads((ROOT/'selection.json').read_bytes())['slots']
 reuse=json.loads((ROOT/'reuse-lock.json').read_bytes());expected={s['slot_id']:s for s in slots};rows=[];run=ROOT/'runs';run.mkdir(exist_ok=False)
 assert sha(PREVIOUS/'freeze.json')==reuse['source_interface_freeze_sha256']
 prior_proof=json.loads((ROOT/'reuse-proof.json').read_bytes())
 assert prior_proof['source_sha256']==reuse['prior_independent_proof_sha256'] and prior_proof['proof']['passed'] and prior_proof['proof']['native_replays']==8
 for f in reuse['files']:
  source=PREVIOUS/f['path'];dest=ROOT/f['path'];assert source.stat().st_size==f['bytes'] and sha(source)==f['sha256']
  dest.parent.mkdir(exist_ok=True,parents=True);shutil.copyfile(source,dest);assert sha(dest)==f['sha256']
 for r in reuse['rows']:rows.append(dict(r,stratum=expected[r['slot_id']]['stratum'],reused_from=PREVIOUS.name))
 put(run/'full-started.json',dict(pid=os.getpid(),time_unix=time.time(),admission=admission,reused_slots=[r['slot_id'] for r in rows],new_slot_count=92,arm='N'))
 def fixed():
  by={r['slot_id']:r for r in rows};return [by.get(s['slot_id'],dict(s,legal_completion=False,stop_reason='not_started')) for s in slots]
 try:
  study,cases,rollout=prepare();by={c['source_id']:c for c in cases};backend=EvaluationBackend(admission);backend._ensure_loaded();put(run/'full-model-identity.json',backend.identity)
  for slot in pending(slots,rows):
   if time.time()>=admission['gpu_deadline_unix']-20:raise BudgetExhausted('original deadline exhausted')
   case=by[slot['source_id']];folder=run/slot['slot_id'];folder.mkdir(exist_ok=False);adapter=interface.RecoveryAdapter(backend)
   bridge=ContextBackend(case,adapter,'N',folder,admission);row=dict(slot,legal_completion=False)
   try:
    trace=rollout(case['item'],study,bridge,case['snapshot']);put(folder/'trace.json',trace)
    replay=CPUReplay(bridge.records);again=rollout(case['item'],study,replay,case['snapshot']);exact=raw_bytes(trace)==raw_bytes(again) and replay.position==len(bridge.records)
    if not exact or not bridge.target_exact:raise TechnicalFailure('native replay or initial target differs')
    row.update(legal_completion=bool(trace['terminated']),native_replay_exact=exact,actual_target_exact=True,prediction=trace['final_result'],stop_reason=trace['stop_reason'],executed_steps=trace['new_steps'],planner_attempts=len(trace['planner_attempts']))
   except (ResourceFailure,TechnicalFailure,BudgetExhausted,OSError) as exc:
    row.update(stop_reason='technical_failure',error_type=type(exc).__name__,error=str(exc));raise
   except Exception as exc:row.update(stop_reason='slot_error',error_type=type(exc).__name__,error=str(exc))
   finally:
    row['model_calls']=len(bridge.records);put(folder/'result.json',row);rows.append(row)
  put(run/'completion.json',dict(stage='N100_complete',source_denominator=100,slot_denominator=100,strata={'train_isolated':46,'train_seen':54},arm='N',reused_slots=8,new_slots=92,rows=fixed(),new_model_calls=sum(r['model_calls'] for r in rows if 'reused_from' not in r),reused_model_calls=85,time_unix=time.time()))
  return 0
 except BaseException as exc:
  put(run/'full-failure.json',dict(error_type=type(exc).__name__,error=str(exc),rows=fixed(),denominator=100,time_unix=time.time()));raise
if __name__=='__main__':raise SystemExit(main())
