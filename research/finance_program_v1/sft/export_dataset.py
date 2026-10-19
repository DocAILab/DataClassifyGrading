"""Export only frozen, actually executed initial Planner supervision; no labels."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


ROOT=Path(__file__).resolve().parent
TEACHER=ROOT.parent/'finance_program_teacher_balanced_20261016_r9/reviews/final-01'
DELIVERY_SHA='cc9609e98a6b11a4cd0355cef132949d2bb3b171887883b826b31ad751cc8412'


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)


def validate_selection(ids,reserve,expected=200):
    if (len(ids)!=expected or len(set(ids))!=expected or len(set(reserve))!=len(reserve)
            or set(ids).intersection(reserve) or any(not isinstance(x,str) or not x for x in ids+reserve)):
        raise ValueError('frozen unique selection or reserve boundary differs')


def reject_label_keys(value):
    if isinstance(value,dict):
        if set(value).intersection({'gold','classification','grading','correct_answer','expected_label','evaluation_label'}):
            raise ValueError('label-bearing input is forbidden')
        for v in value.values():reject_label_keys(v)
    elif isinstance(value,list):
        for v in value:reject_label_keys(v)


def extract_example(trace,source_id,trace_sha):
    if trace.get('terminated') is not True or len(trace.get('planner_attempts',[]))!=1 or len(trace.get('programs',[]))!=1:
        raise ValueError('this frozen dataset contains only complete initial single-Program trajectories')
    a=trace['planner_attempts'][0];messages=a.get('messages');response=a.get('raw_response')
    if (not isinstance(messages,list) or len(messages)!=2 or [m.get('role') for m in messages]!=['system','user']
            or any(not isinstance(m.get('content'),str) or not m['content'] for m in messages)
            or not messages[0]['content'].startswith('You are the Global Program Planner')):
        raise ValueError('Planner inputs are not the original two-message initial interface')
    payload=json.loads(messages[1]['content']);reject_label_keys(payload)
    library=payload.get('skill_library',[]);names=[s['name'] for s in library]
    if len(names)!=66 or len(set(names))!=66:raise ValueError('complete F66 contracts required')
    plan=json.loads(response)
    if (set(plan)!= {'program'} or not isinstance(plan['program'],list) or not plan['program']
            or any(set(s)!= {'tool','arguments'} or s['tool'] not in names or not isinstance(s['arguments'],dict) for s in plan['program'])
            or plan!=trace['programs'][0]['plan'] or not trace['programs'][0].get('terminated')):
        raise ValueError('raw response differs from actual executed Program')
    return {'source_id':source_id,'kind':'initial','messages':messages,'response':response,'trace_sha256':trace_sha}


def main():
    final=TEACHER/'final-review-01';selection=read(final/'final-set.json');reviews=read(final/'manual-final-review.json')
    validate_selection(selection['training_ids'],selection['reserve_ids'])
    if not selection['final_review_complete'] or not selection['delivery_ready'] or selection['hard_evidence_faults']:
        raise ValueError('final teacher evidence not ready')
    if sha(final/'teacher-200-delivery-01.tar.gz')!=DELIVERY_SHA:raise ValueError('teacher delivery identity changed')
    all_rows={r['id']:r for r in read(TEACHER/'combined-admission.json')['rows']};rows=[];provenance=[];classes=Counter()
    for source_id in selection['training_ids']:
        r=all_rows[source_id];review=reviews[source_id];p=Path(r['trace_path'])
        if not r['accepted'] or review['reviewed'] is not True or review['serious_issue'] or sha(p)!=r['trace_sha256'] or review['trace_sha256']!=r['trace_sha256']:
            raise ValueError('selected source review/trace changed')
        t=read(p);rows.append(extract_example(t,source_id,r['trace_sha256']));classes[t['final_result']['category']]+=1
        provenance.append({'source_id':source_id,'trace_path':str(p),'trace_sha256':r['trace_sha256'],'source_batch':r['source_batch'],
                           'warnings':r['warnings'],'messages_sha256':hashlib.sha256(json.dumps(rows[-1]['messages'],ensure_ascii=False,sort_keys=True).encode()).hexdigest(),
                           'response_sha256':hashlib.sha256(rows[-1]['response'].encode()).hexdigest()})
    folder=ROOT/'data';folder.mkdir(exist_ok=False);data=folder/'planner200.jsonl'
    with data.open('x',encoding='utf-8',newline='\n') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n')
    save(ROOT/'reviews/dataset-provenance-01.json',{'rows':provenance,'teacher_delivery_sha256':DELIVERY_SHA,
        'selection_sha256':sha(final/'final-set.json'),'review_sha256':sha(final/'manual-final-review.json')})
    report={'rows':len(rows),'unique_sources':len({r['source_id'] for r in rows}),'reserve_sources_not_used':len(selection['reserve_ids']),
            'data_sha256':sha(data),'data_bytes':data.stat().st_size,'kinds':{'initial':len(rows),'recovery':0},'category_counts_local_only':dict(classes),
            'source_labels_in_model_input':False,'future_observations_in_initial_input':False,'raw_reasoning_supervised':False,
            'skill_responses_supervised':False,'source_equally_weighted':True,'student_training_started':False}
    save(ROOT/'reviews/dataset-export-01.json',report);print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
