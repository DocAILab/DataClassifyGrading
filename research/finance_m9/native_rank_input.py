"""No filesystem/cohort access: convert only this native SEARCH request's acquired facts."""
from copy import deepcopy
import json
def make_rank_messages(search_messages,registry,templates):
 if len(search_messages)!=2 or [x['role'] for x in search_messages]!=['system','user'] or not search_messages[0]['content'].startswith('Execute only the selected Skill SEARCH_STANDARD_'):raise ValueError('native SEARCH request required')
 u=json.loads(search_messages[1]['content'])
 if any(k in u for k in ['target_option_id','reference_label','ground_truth','classification']):raise ValueError('reference material forbidden')
 options=u['category_options'];fields=['option_id','label','description']
 if len(options)!=12 or u['top_k']!=5 or [[x[k] for k in fields] for x in options]!=[[x[k] for k in fields] for x in registry]:raise ValueError('unchanged full public pool required')
 boundary=deepcopy(u['evidence_boundary']);metadata={}
 allowed={'database_name','database_description','table_name','table_description','field_name','field_description','field_type','value'}
 for fact_id,fact in boundary['facts'].items():
  if fact_id.startswith('metadata.') and fact_id[9:] in allowed:
   if fact.get('role')!='raw_fact' or fact.get('source_scope')!='target_record':raise ValueError('non-target metadata is not an acquired source fact')
   if isinstance(fact['value'],str) and fact['value'].strip():metadata[fact_id[9:]]=fact['value']
 record=u['record_input'];available={'domain':record['domain'],'field_name':record['field_name'],'metadata':metadata}
 body={'all_available_evidence':available,'available_evidence':available,'candidate_details':deepcopy(options),'evidence_boundary':boundary,'evidence_gap_ledger':deepcopy(u.get('evidence_gap_ledger',[])),'output_extension':'support is required; only evidence_boundary.facts may be cited','output_schema':{'option_id':'one supplied short option ID or UNRESOLVED','summary':'short string; at most 20 characters','support':{'sufficient':'boolean','citations':[{'fact_id':'exact ID from evidence_boundary.facts','quote':'exact raw fact quote'}],'missing_evidence':'string; explain decision-critical gaps when insufficient'}}}
 return [{'role':'system','content':templates['classification_system']},{'role':'user','content':json.dumps(body,ensure_ascii=False,sort_keys=True,separators=(',',':'))}]
def rank_reply(ids):
 if len(ids)!=5 or len(set(ids))!=5:raise ValueError('unique Top5 required')
 return json.dumps({'option_ids':ids,'summary':'概率排序候选'},ensure_ascii=False,separators=(',',':'))
