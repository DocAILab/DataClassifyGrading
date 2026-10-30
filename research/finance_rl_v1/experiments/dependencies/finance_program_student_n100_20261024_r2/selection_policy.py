"""Label-free fixed100 N-only:46 train-isolated groups and54 train-seen groups."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent
def raw(v):return json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
def group(row):return hashlib.sha256(raw(tuple(row['metadata'][k] for k in ('database_name','table_name','field_description')))).hexdigest()
def build():
 origin=BASE/'finance_program_student_r9_eval80_20261022_r1'
 rows=json.loads((origin/'inputs.json').read_bytes())['study']['rows'];old=json.loads((origin/'selection.json').read_bytes())
 train=json.loads((BASE/'finance_program_student_fresh162_20261022_r1/data/data-lock.json').read_bytes())['identities']
 ts={r['source_id'] for r in train};tg={r['group'] for r in train};excluded=set(old['diagnostic_source_ids'])|set(old['reserve_excluded'])
 blocked={group(r) for r in rows if r['id'] in excluded};eligible={};seen={}
 isolated_key=lambda r:hashlib.sha256(raw([20261022,'fresh162-eval',r['id']])).hexdigest()
 seen_key=lambda r:hashlib.sha256(raw([20261024,'N100-seen',r['id']])).hexdigest()
 for r in sorted(rows,key=isolated_key):
  g=group(r)
  if r['id'] in ts|excluded or g in tg|blocked:continue
  eligible.setdefault(g,r)
 for r in sorted(rows,key=seen_key):
  g=group(r)
  if r['id'] not in ts or g not in tg or r['id'] in excluded or g in blocked:continue
  seen.setdefault(g,r)
 assert len(eligible)==46 and len(seen)>=54
 selected=[(r,'train_isolated') for r in eligible.values()]+[(r,'train_seen') for r in list(seen.values())[:54]]
 slots=[dict(slot_id=f'{i:03d}-N',source_id=r['id'],source_index=i,arm='N',group=group(r),stratum=stratum) for i,(r,stratum) in enumerate(selected,1)]
 return dict(slots=slots,source_denominator=100,slot_denominator=100,arm='N',train_isolated=46,train_seen=54,labels_used=False,unique_sources=len({s['source_id'] for s in slots}),unique_groups=len({s['group'] for s in slots}),reference_timing='local_posthoc_only')
