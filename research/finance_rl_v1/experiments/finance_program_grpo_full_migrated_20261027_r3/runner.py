"""All48 current-policy sweep with audited episode reuse and checkpoint migration."""
import json,os,time,hashlib
from pathlib import Path
from support import ROOT,put,verify,digest,sha
from full_contract import groups,gate
from recovery import preserve_incomplete,require_hash
from episode_runtime import episode

def read(path):return json.loads(Path(path).read_bytes())
def preserve(folder,attempt):
 if not folder.exists():return
 target=preserve_incomplete(folder,ROOT/'runs/interrupted',attempt)
 if target:
  put(ROOT/f'runs/relocations/{time.time_ns()}.json',dict(old=folder.relative_to(ROOT).as_posix(),preserved=target.relative_to(ROOT).as_posix(),attempt=attempt,files=[dict(path=p.relative_to(target).as_posix(),sha256=sha(p)) for p in sorted(target.rglob('*')) if p.is_file()]))

def main():
 verify();attempt=int(os.environ['FINANCE_ATTEMPT_ID']);admission=read(ROOT/f'attempts/{attempt:03d}/admission.json')
 from native_env import Environment
 from backend import make_backend,planner_turns
 from policy_update import update
 from checkpoint_store import save,restore
 public=read(ROOT/'cohort-public.json');selection=read(ROOT/'selection-proof.json');labels=read(ROOT/'reward-labels.json');env=Environment(public)
 by={c['source_id']:c for c in env.cases};assert len(by)==48 and len(selection['batches'])==6
 run=ROOT/'runs';run.mkdir(exist_ok=True);backend=make_backend(admission);rows=[];batches=[];version=0
 try:
  backend._ensure_loaded();torch=backend.runtime['torch']
  backend.optimizer=torch.optim.AdamW([p for _,p in backend.named],lr=5e-6,betas=(.9,.999),eps=1e-8,weight_decay=0.)
  for bi in range(1,7):
   path=run/f'batches/batch-{bi:02d}-completion.json'
   if not path.exists():break
   b=read(path);assert b['input_policy_version']==version;batches.append(b);version=b['output_policy_version'];rows.extend(b['rows'])
  ck=run/f'checkpoints/version-{version:02d}'
  if not ck.exists():
   if version:raise ValueError('committed policy checkpoint missing')
   save(ck,backend.named,backend.optimizer,version=0,reference_digest=backend.reference_digest)
  restored=restore(ck,backend.named,backend.optimizer,reference_digest=backend.reference_digest);backend.version=version;backend.freeze_for_generation()
  put(run/f'resumes/{attempt:03d}.json',dict(attempt=attempt,policy_version=version,checkpoint_sha256=restored['state_sha256'],actual_tensor_digest=digest(backend.weight_shas),reference_digest=backend.reference_digest,completed_batches=len(batches),reused_episodes=len(rows),actual_CUDA_RNG_restore=True,actual_restore_validation=restored.get('actual_restore_validation'),time_unix=time.time()))
  for bi,ids in enumerate(selection['batches'],1):
   if bi<=len(batches):continue
   fresh=[]
   for ci,sid in enumerate(ids,1):
    for sample in range(4):
     folder=run/f'episodes/b{bi}-{ci:02d}-{sample+1}';result=folder/'result.json';receipt=run/f'episode-receipts/b{bi}-{ci:02d}-{sample+1}.json'
     if result.exists():
      require_hash(result,read(receipt)['result_sha256']);row=read(result)
      if row['policy_version']!=version or row['source_id']!=sid or row['sample_index']!=sample:raise ValueError('stale completed episode cannot be reused')
     else:
      if folder.exists():preserve(folder,attempt)
      row=episode(env,backend,by[sid],folder,202610270000+bi*1000+ci*101+sample,sample,labels[sid])
      put(receipt,dict(result_sha256=sha(result),attempt=attempt,policy_version=version,source_id=sid,sample_index=sample))
     rows.append(row);fresh.append(row)
   admitted=gate(fresh,version);rollout=run/f'batches/batch-{bi:02d}-rollouts.json'
   value=dict(policy_version=version,source_ids=ids,gate=admitted,rows=fresh)
   if rollout.exists():assert read(rollout)==value
   else:put(rollout,value)
   if admitted['allow_update']:
    if version==0 and not (run/'cached-probability-replay.json').exists():
     turns=[t for v in fresh for t in planner_turns(ROOT/v['folder'])];backend.cached_probability_replay(turns[0])
    step=run/f'updates/step-{version+1:02d}'
    if (step/'completion.json').exists():
     u=read(step/'completion.json');restore(run/f'checkpoints/version-{version+1:02d}',backend.named,backend.optimizer,reference_digest=backend.reference_digest);backend.version=version+1;backend.freeze_for_generation()
    else:
     if (run/f'checkpoints/version-{version+1:02d}/manifest.json').exists():raise ValueError('unreceipted committed checkpoint requires CPU diagnosis, no repeated optimizer')
     if step.exists():preserve(step,attempt)
     starts=list(run.glob('updates/*/optimizer-start.json'))+list((run/'interrupted').glob('step-*/optimizer-start.json'))
     if len(starts)>=8:raise ValueError('global optimizer attempt limit')
     u=update(backend,groups(fresh,version),'A1')
   else:u=dict(updated=False,reason='no_outcome_variation_or_legal_gate')
   b=dict(batch=bi,input_policy_version=version,output_policy_version=backend.version,source_ids=ids,gate=admitted,update=u,rows=fresh)
   put(run/f'batches/batch-{bi:02d}-completion.json',b);batches.append(b);version=backend.version
  assert len(rows)==192 and len(batches)==6
  verify();put(run/'completion.json',dict(completed=True,kind='full48_Planner_GRPO',targets=48,episodes=192,rows=rows,batches=batches,updates=[b['update'] for b in batches],actual_optimizer_updates=backend.version,reward='A1',total_claims=len(list((ROOT/'campaign/claims').glob('*.json'))),reference_digest=backend.reference_digest,actor_tensor_digest=digest(backend.weight_shas),no_update_reason=None if backend.version else 'no_nonzero_result_advantages',fresh_test=False,time_unix=time.time()))
  return 0
 except BaseException as exc:
  put(ROOT/f'attempts/{attempt:03d}/failure.json',dict(type=type(exc).__name__,error=str(exc),actual_optimizer_updates=backend.version,completed_rows=len(rows),time_unix=time.time()))
  raise
if __name__=='__main__':raise SystemExit(main())
