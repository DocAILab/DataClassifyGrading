"""Prepare original162 text plus label-free disjoint public metadata evaluation."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from contract import ROOT,SEED,put,sha,raw,require_isolated
PREP=ROOT.parent/'finance_program_student_merge162_local_20261022_r1'
sys.path.insert(0,str(PREP))
import importlib.util
spec=importlib.util.spec_from_file_location('sealed162_preparation',PREP/'prepare.py')
previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)

def group(row):
    return hashlib.sha256(raw(tuple(row['metadata'][k] for k in ('database_name','table_name','field_description')))).hexdigest()
def main():
    rows,_,refs=previous.load_bundle()
    ordered=sorted(rows,key=lambda r:hashlib.sha256(raw([SEED,'fresh162-train',r['id']])).hexdigest())
    data=ROOT/'data';data.mkdir(exist_ok=False)
    train=data/'train162.jsonl'
    with train.open('xb') as f:
        for r in ordered:f.write(json.dumps({k:r[k] for k in ('id','messages','completion')},ensure_ascii=False,separators=(',',':')).encode()+b'\n')
    origin=ROOT.parent/'finance_program_student_r9_eval80_20261022_r1'
    old=json.loads((origin/'inputs.json').read_bytes());s=json.loads((origin/'selection.json').read_bytes())
    train_sources={r['source_id'] for r in rows};train_groups={r['group'] for r in rows}
    excluded=set(s['diagnostic_source_ids'])|set(s['reserve_excluded'])
    blocked_groups={group(r) for r in old['study']['rows'] if r['id'] in excluded}
    eligible={}
    key=lambda r:hashlib.sha256(raw([SEED,'fresh162-eval',r['id']])).hexdigest()
    for r in sorted(old['study']['rows'],key=key):
        g=group(r)
        if r['id'] in train_sources|excluded or g in train_groups|blocked_groups:continue
        eligible.setdefault(g,r)
    selected=[dict(source_id=r['id'],group=g) for g,r in sorted(eligible.items(),key=lambda x:key(x[1]))[:40]]
    if len(selected)!=40:raise ValueError('fewer than fixed40 isolated groups')
    require_isolated(rows,selected)
    study=deepcopy(old['study']);by_id={r['id']:r for r in study['rows']};cases=[];slots=[]
    for i,r in enumerate(selected):
        sid=r['source_id'];source=by_id[sid];field=source['metadata']['field_name']
        item=dict(id='fresh-initial-'+sid,source_id=sid,kind='initial',initial=dict(source_id=sid,domain=source['domain'],field_name=field))
        snapshot=dict(state={'availability_ledger':[dict(scope='record',target=field,type='field_name',status='obtained',skill='initial-input',source='public-input',value=field)]},
            counts={},lineage=dict(evidence=0,candidates=0,selection=0),comparison_entries={},prefix_steps=[],feedback=None,target_exact=True,prefix_calls=0)
        cases.append(dict(source_id=sid,item=item,snapshot=snapshot,prefix=[],kind='initial'))
        for arm in (('F','N') if i%2==0 else ('N','F')):
            slots.append(dict(slot_id=f'{i+1:03d}-{arm}',source_id=sid,source_index=i+1,arm=arm,group=r['group']))
    put(ROOT/'eval-inputs.json',dict(study=study,cases=cases))
    put(ROOT/'selection.json',dict(selected=selected,slots=slots,eligible_groups=len(eligible),labels_used=False,train_source_overlap=0,train_group_overlap=0))
    identities=[dict(id=r['id'],source_id=r['source_id'],group=r['group'],supervision_state=r['supervision_state'],
                    messages_sha256=hashlib.sha256(json.dumps(r['messages'],ensure_ascii=False,separators=(',',':')).encode()).hexdigest(),
                    completion_sha256=hashlib.sha256(r['completion'].encode()).hexdigest()) for r in ordered]
    put(data/'data-lock.json',dict(train_sha256=sha(train),rows=162,identities=identities,merge_inputs=refs,
        evaluation_inputs_sha256=sha(ROOT/'eval-inputs.json'),selection_sha256=sha(ROOT/'selection.json'),old200_training_rows=0))
    print(json.dumps(dict(rows=162,initial=104,conditional=58,eval_groups=40,eligible_groups=len(eligible),train_sha256=sha(train),overlap=0)))
if __name__=='__main__':main()
