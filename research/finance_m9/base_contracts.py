"""Fixed two-Skill development comparison; no reference labels in requests."""
from copy import deepcopy
import json
from retrieval_input_contract import disclose_descriptions

def dumps(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'))
def strict_json(raw):
    def pairs(xs):
        out={}
        for k,v in xs:
            if k in out:raise ValueError('duplicate JSON key')
            out[k]=v
        return out
    try:return json.loads(raw,object_pairs_hook=pairs)
    except (TypeError,ValueError) as exc:raise ValueError('strict JSON object required') from exc

def evidence(case):
    m=deepcopy(case['metadata'])
    allowed={'database_name','database_description','table_name','table_description','field_name','field_description','field_type','value'}
    if not isinstance(m,dict) or not set(m)<=allowed:raise ValueError('whitelisted metadata only')
    if not all(isinstance(v,str) and v.strip() for v in m.values()):raise ValueError('observed nonempty source strings required')
    facts={'metadata.'+k:{'value':v,'provenance':'resident-original-metadata','decisive_eligible':k in {'field_description','table_description','database_description','value'},'role':'raw_fact','source_scope':'target_record'} for k,v in m.items()}
    return {'domain':'finance','field_name':m['field_name'],'metadata':m}, {'schema':'acquired-evidence-boundary-v1','facts':facts,'hypotheses':{},'authority':'Actual source metadata values only. A valid literal quote is not proof that a category inference follows. Category descriptions are standards, not source business facts.'}

def make_search(case,registry,templates,arm):
    if arm not in {'labels_only','with_descriptions'}:raise ValueError('unknown fixed arm')
    available,boundary=evidence(case);m=available['metadata']
    u={'record_input':{'domain':'finance','field_name':m['field_name']},'semantic_query':{'schema':'grounded-source-v1','status':'observed_description','field_name':m['field_name'],'description':m['field_description'],'source_consistency':'not_adjudicated'},'top_k':5,'category_options':[{k:x[k] for k in ['option_id','label']} for x in registry],'available_evidence':available,'evidence_boundary':boundary,'evidence_gap_ledger':[]}
    messages=[{'role':'system','content':templates['search_system']},{'role':'user','content':dumps(u)}]
    return disclose_descriptions(messages,registry) if arm=='with_descriptions' else messages

def make_classification(case,registry,templates,ids):
    if len(ids)!=5 or len(set(ids))!=5:raise ValueError('exact parsed Top-5 required')
    by={x['option_id']:x for x in registry}
    if any(x not in by for x in ids):raise ValueError('candidate outside registry')
    available,boundary=evidence(case)
    u={'all_available_evidence':available,'candidate_details':[deepcopy(by[x]) for x in ids],'available_evidence':available,'evidence_boundary':boundary,'output_extension':'support is required; only evidence_boundary.facts may be cited','evidence_gap_ledger':[],
       'output_schema':{'option_id':'one supplied short option ID or UNRESOLVED','summary':'short string; at most 20 characters','support':{'sufficient':'boolean','citations':[{'fact_id':'exact ID from evidence_boundary.facts','quote':'exact raw fact quote'}],'missing_evidence':'string; explain decision-critical gaps when insufficient'}}}
    return [{'role':'system','content':templates['classification_system']},{'role':'user','content':dumps(u)}]

def parse_retrieval(raw,registry):
    x=strict_json(raw)
    if not isinstance(x,dict) or set(x)!={'option_ids','summary'}:raise ValueError('retrieval output keys')
    ids=x['option_ids'];allowed={v['option_id'] for v in registry}
    if not isinstance(ids,list) or len(ids)!=5 or not all(isinstance(v,str) for v in ids):raise ValueError('five string option IDs required')
    if len(set(ids))!=5 or any(v not in allowed for v in ids):raise ValueError('unique supplied options required')
    if not isinstance(x['summary'],str) or not x['summary'].strip():raise ValueError('nonempty summary required')
    return x

def parse_classification(raw,user):
    x=strict_json(raw)
    if not isinstance(x,dict) or set(x)!={'option_id','summary','support'}:raise ValueError('classification output keys')
    o=x['option_id'];s=x['support']
    if not isinstance(o,str) or not isinstance(x['summary'],str) or not x['summary'].strip():raise ValueError('typed ID and summary required')
    if not isinstance(s,dict) or set(s)!={'sufficient','citations','missing_evidence'} or type(s['sufficient']) is not bool:raise ValueError('typed support required')
    if not isinstance(s['citations'],list) or not isinstance(s['missing_evidence'],str):raise ValueError('typed support members')
    if s['sufficient']:
        if o not in {v['option_id'] for v in user['candidate_details']} or not s['citations'] or s['missing_evidence']:raise ValueError('sufficient choice must cite actual selected option')
        for c in s['citations']:
            if not isinstance(c,dict) or set(c)!={'fact_id','quote'} or not isinstance(c['fact_id'],str):raise ValueError('typed citation')
            f=user['evidence_boundary']['facts'].get(c['fact_id'])
            if not f or not f['decisive_eligible'] or not isinstance(c['quote'],str) or not c['quote'] or c['quote'] not in f['value']:raise ValueError('invalid raw quote')
    elif o!='UNRESOLVED' or s['citations'] or not s['missing_evidence'].strip():raise ValueError('explicit unresolved distinction required')
    return x
