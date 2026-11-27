import hashlib,json
KEYS={'database_name','database_description','table_name','table_description','field_name','field_description','field_type','value'}
def clean_metadata(metadata):return {k:v for k,v in metadata.items() if k in KEYS and isinstance(v,str) and v.strip() and v.strip().lower() not in ('unknown','none','null','nan')}
def digest(metadata):return hashlib.sha256(json.dumps(metadata,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def project_rows(rows,selectors):
 lookup={r['id']:r for r in rows}
 if len(lookup)!=len(rows) or len({x['source_id'] for x in selectors})!=len(selectors):raise ValueError('duplicate source/selector')
 out=[]
 for item in selectors:
  if set(item)!={'source_id','metadata_sha256','case_index','arm_order'} or item['source_id'] not in lookup:raise ValueError('explicit metadata selector required')
  metadata=clean_metadata(lookup[item['source_id']]['metadata'])
  if digest(metadata)!=item['metadata_sha256']:raise ValueError('resident metadata SHA mismatch')
  if not metadata.get('field_name') or not metadata.get('field_description'):raise ValueError('declared complete VAL422 metadata was not actually present')
  out.append({**item,'metadata':metadata})
 return out
