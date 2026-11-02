"""Fixed Val422 cohort and label-free inputs; no model or reference-label access."""
from collections import Counter
from copy import deepcopy
import hashlib
import json


def digest(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode('utf-8')).hexdigest()


def no_labels(value):
    if isinstance(value,dict):
        for k,v in value.items():
            if k.lower() in {'ground_truth','ground_truth_level','classification','grading','category','category_id','grading_level','label','labels','label_status'}:
                raise ValueError('label in public metadata: '+k)
            no_labels(v)
    elif isinstance(value,list):
        for v in value:no_labels(v)


def prepare(original,train,diagnostic):
    rows=deepcopy(original['rows'])
    no_labels(rows)
    ids=[r['id'] for r in rows]
    if len(ids)!=422 or len(set(ids))!=422:raise ValueError('exact unique422 required')
    trained={r['source_id'] for r in train}
    diag={r['source_id'] for r in diagnostic}
    if set(ids)&trained or set(ids)&diag:raise ValueError('source IDs overlap train or100')
    train_groups={r['group'] for r in train}
    cases=[];slots=[];scopes=[]
    group_overlap=[]
    for i,r in enumerate(rows,1):
        sid=r['id'];md=r['metadata'];field=md['field_name']
        if not isinstance(sid,str) or any(c in sid for c in '/\\:') or sid in ('.','..'):raise ValueError('unsafe source id')
        group=digest([md.get(k,'') for k in ('database_name','table_name','field_description')])
        overlap=group in train_groups
        if overlap:group_overlap.append(sid)
        slot=dict(slot_id=f'{i:03d}-N',source_id=sid,source_index=i,arm='N',group=group,
                  stratum='train_semantic_group_overlap' if overlap else 'train_source_and_group_isolated')
        initial=dict(source_id=sid,domain=r['domain'],field_name=field)
        snapshot=dict(state={'availability_ledger':[dict(scope='record',target=field,type='field_name',status='obtained',skill='initial-input',source='public-input',value=field)]},
                      counts={},lineage=dict(evidence=0,candidates=0,selection=0),comparison_entries={},prefix_steps=[],feedback=None,target_exact=True,prefix_calls=0)
        cases.append(dict(source_id=sid,item=dict(id='fresh-initial-'+sid,source_id=sid,kind='initial',initial=initial),snapshot=snapshot,prefix=[],kind='initial'))
        slots.append(slot)
        db=md.get('database_name','');table=md.get('table_name','')
        scopes.append(dict(id=sid,domain=r['domain'],source_database_name=db,source_table_name=table,
                           source_table_id='source-table-'+digest([r['domain'],db,table])[:20],metadata=deepcopy(md)))
    study=dict(rows=rows,registry=deepcopy(original['registry']),budget=deepcopy(original['budget']))
    proof=dict(rows=422,source_train_overlap=0,diagnostic100_overlap=0,train_semantic_group_overlap=len(group_overlap),
               train_semantic_group_overlap_source_ids=group_overlap,labels_used=False,order='original_full422_metadata_order',
               N_only=True,reused_100_results=0)
    return dict(study=study,cases=cases,slots=slots,scopes=scopes,proof=proof)


def summarize(rows):
    return dict(denominator=422,observed=len(rows),missing=422-len(rows),
                algorithm_completed=sum(r.get('algorithm_completed',False) for r in rows),
                predictions=sum(r.get('legal_completion',False) and r.get('prediction') is not None for r in rows),
                abstentions=sum(r.get('stop_reason')=='abstained_terminal' for r in rows),
                stop_reasons=dict(Counter(r.get('stop_reason','missing') for r in rows)))
