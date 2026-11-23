from copy import deepcopy
import hashlib,json
def digest(value):return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def build_public(template,rows):
 study=template['inputs']['study'];cases=[];scopes=[];records=[]
 for row in rows:
  sid=row['source_id'];m=deepcopy(row['metadata']);field=m['field_name']
  scope_key=['known_exact_table',m.get('database_name',''),m['table_name']] if m.get('database_name') and m.get('table_name') else ['no_table_per_record',sid]
  family=digest(scope_key)
  records.append({'id':sid,'domain':'finance','metadata':m})
  scopes.append({'id':sid,'domain':'finance','metadata':deepcopy(m),'source_database_name':m.get('database_name',''),'source_table_name':m.get('table_name',''),'source_table_id':'source-table-'+family[:20]})
  cases.append({'kind':'initial','source_id':sid,'family':family,'prefix':[],'item':{'id':'M9-eval-'+sid,'source_id':sid,'kind':'initial','initial':{'domain':'finance','field_name':field,'source_id':sid}},'snapshot':{'comparison_entries':{},'counts':{},'feedback':None,'lineage':{'candidates':0,'evidence':0,'selection':0},'prefix_calls':0,'prefix_steps':[],'state':{'availability_ledger':[{'scope':'record','skill':'initial-input','source':'public-input','status':'obtained','target':field,'type':'field_name','value':field}]},'target_exact':True}})
 return {'inputs':{'study':{'budget':deepcopy(study['budget']),'registry':deepcopy(study['registry']),'rows':records},'cases':cases},'scopes':scopes}
