"""Prospective original-training metadata acquisition only; no teacher-only facts."""
import copy,hashlib,importlib.util,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PARENT=ROOT.parent/'finance_teacher_200_controller_20261024_r10'
SCOPE=ROOT.parent/'finance_program_recovery_source_scope_20261024_r8/source_scope.py'
def read(p):return json.loads(Path(p).read_bytes())
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
scope=load('r12_original_scope_contract',SCOPE)
def fixture():
 old=read(PARENT/'program-inputs.json');scopes=read(ROOT.parent/'finance_program_recovery_source_scope_20261024_r9/source-scopes.json')['rows'];raw=read(ROOT.parent/'finance_sft_v2_20261001/raw/DCG_FIN_SFT.json');origin={r['id']:r for r in raw};schedule=read(ROOT.parent/'finance_teacher_same_information_20261024_r1/schedule.json');dev={i for i,v in schedule['source_partition'].items() if v['split']!='train'}
 data=ROOT.parent/'finance_sft_v2_20261001';training=read(data/'data/metadata-sft.json');manifest=read(data/'data_manifest.json');split=read(data/'data/split_boundaries.json')
 if hashlib.sha256((data/'data/metadata-sft.json').read_bytes()).hexdigest()!=manifest['files']['data/metadata-sft.json'] or hashlib.sha256((data/'raw/DCG_FIN_SFT.json').read_bytes()).hexdigest()!=split['split_sha256']['sft']:raise ValueError('original SFT source identity differs')
 rows=old['study']['rows'];expanded=copy.deepcopy(rows);extended=copy.deepcopy(scopes);by={r['id']:r for r in rows};scopeby={r['id']:r for r in scopes};targetids={c['source_id'] for c in old['cases']}
 if set(by)!=set(scopeby):raise ValueError('old source scope mismatch')
 def named(v):return scope.native.present(v)
 devgroups={origin[i].get('source_table_id') for i in dev if i in origin};devtables={origin[i]['metadata'].get('table_name') for i in dev if i in origin and named(origin[i]['metadata'].get('table_name'))}
 heldout=set(split['ids']['val'])|set(split['ids']['test'])
 def same(target,r):
  a=target['metadata'];b=r['metadata']
  if target['domain']!=r['domain'] or not named(a.get('table_name')):return False
  if named(a.get('database_name')):return a['database_name']==b.get('database_name') and a['table_name']==b.get('table_name')
  return scopeby[target['id']]['source_table_id']==origin[r['id']]['source_table_id'] and a['table_name']==b.get('table_name')
 targets=[by[i] for i in sorted(targetids)];added=[]
 for r in sorted(training,key=lambda r:r['id']):
  i=r['id'];o=origin[i]
  if i in by or i in dev or i in heldout or o.get('label_status')!='original' or o.get('source_table_id') in devgroups or r['metadata'].get('table_name') in devtables:continue
  if r['metadata']!=o['metadata']:raise ValueError('extra source public metadata altered')
  matched=[t for t in targets if same(t,r)]
  if not matched:continue
  # Do not silently replace an already-visible field's old description/value.
  if any(same(t,v) and v['metadata']['field_name']==r['metadata']['field_name'] for t in matched for v in rows):continue
  expanded.append(copy.deepcopy(r));extended.append({k:copy.deepcopy(o[k]) for k in ('id','domain','metadata','source_table_id','source_database_name','source_table_name')});added.append(i)
 scope.SourceScopedMetadata(expanded,extended)
 return dict(rows=rows,scopes=scopes,cases=old['cases'],study=old['study'],expanded_rows=expanded,expanded_scopes=extended,added_ids=added,origin_status={i:r['label_status'] for i,r in origin.items()},reserved_dev_ids=list(dev),new_metadata_only_after_actual_acquisition=True,provided_synthesized_extras_not_added=True)
def lookup(bundle,expanded):return scope.SourceScopedMetadata(bundle['expanded_rows' if expanded else 'rows'],bundle['expanded_scopes' if expanded else 'scopes'])
