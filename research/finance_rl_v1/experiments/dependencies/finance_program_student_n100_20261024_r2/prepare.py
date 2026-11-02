"""Prepare original metadata initial states; no labels or SFT targets sent to model."""
from copy import deepcopy
import json
from common import ROOT,PARENT,PREVIOUS,put,sha
from selection_policy import build
def main():
 selection=build();source=ROOT.parent/'finance_program_student_fresh162_20261022_r1/eval-inputs.json'
 inputs=json.loads(source.read_bytes());study=inputs['study'];by={r['id']:r for r in study['rows']};cases=[]
 old_cases={r['source_id']:r for r in inputs['cases']}
 for s in selection['slots']:
  source_row=by[s['source_id']];field=source_row['metadata']['field_name'];sid=s['source_id']
  item=dict(id='fresh-initial-'+sid,source_id=sid,kind='initial',initial=dict(source_id=sid,domain=source_row['domain'],field_name=field))
  snapshot=dict(state={'availability_ledger':[dict(scope='record',target=field,type='field_name',status='obtained',skill='initial-input',source='public-input',value=field)]},counts={},lineage=dict(evidence=0,candidates=0,selection=0),comparison_entries={},prefix_steps=[],feedback=None,target_exact=True,prefix_calls=0)
  c=dict(source_id=sid,item=item,snapshot=snapshot,prefix=[],kind='initial')
  if sid in old_cases:assert c==old_cases[sid]
  cases.append(c)
 proof=json.loads((PREVIOUS/'reviews/independent-targeted-verification-01.json').read_bytes());assert proof['passed'] and proof['native_replays']==8 and proof['actual_messages']==85
 old_root=PREVIOUS/'reviews/targeted-01/extract';old=json.loads((old_root/'runs/targeted-completion.json').read_bytes());by_slot={s['slot_id']:s for s in selection['slots']};files=[]
 receipt=json.loads((PREVIOUS/'reviews/targeted-01/receipt.json').read_bytes())
 for r in old['rows']:
  wanted=by_slot[r['slot_id']];assert all(r[k]==wanted[k] for k in ('source_id','group','source_index','arm'))
  assert r['native_replay_exact'] and r['actual_target_exact']
  for f in receipt['members']:
   if f['path'].startswith('runs/'+r['slot_id']+'/'):
    p=old_root/f['path'];assert p.stat().st_size==f['bytes'] and sha(p)==f['sha256'];files.append(f)
 put(ROOT/'selection.json',selection);put(ROOT/'eval-inputs.json',dict(study=deepcopy(study),cases=cases))
 put(ROOT/'reuse-lock.json',dict(source_root=PREVIOUS.name,rows=old['rows'],files=files,prior_independent_proof_sha256=sha(PREVIOUS/'reviews/independent-targeted-verification-01.json'),prior_archive_sha256=proof['archive_sha256'],source_interface_freeze_sha256=sha(PREVIOUS/'freeze.json'),reused_model_calls=85,prior_native_exact=8))
 print(json.dumps(dict(prepared=True,slots=100,isolated=46,seen=54,reused=8,new_slots=92,labels_used=False,reused_files=len(files))))
if __name__=='__main__':main()
