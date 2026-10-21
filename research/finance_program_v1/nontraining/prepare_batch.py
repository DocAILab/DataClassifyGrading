"""Validate the frozen Val inputs and mechanically separate private gold."""
# 准备阶段将元数据与私有答案分开保存；后续推理仅使用 study，答案留给离线评分。
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
BUDGET=dict(max_programs=3,max_program_steps=10,max_skill_steps=12,max_planner_attempts=4,max_identical_calls=1)
DATA_SHA={'data/metadata-val.json':'e6792db18afe1ac82a3cae499735182a7bcee366a42251e72a8ccad5e6aee8c5',
          'data/registry.json':'1b19032a3d2d8d283044e05f3f97cab2d4d77d71b68e3fb97e222f75c2705bec'}

def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f: json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)

def validate_study(study):
    if study['budget']!=BUDGET: raise ValueError('fixed Program budget changed')
    ids=[r['id'] for r in study['rows']]
    if len(ids)!=len(set(ids)): raise ValueError('duplicate metadata identity')
    if any(set(r)!={'id','domain','metadata'} for r in study['rows']): raise ValueError('labels in metadata rows')
    selected=[i for group in study['cohorts'].values() for i in group]
    if len(selected)!=len(set(selected)) or not set(selected)<=set(ids): raise ValueError('cohorts overlap or unknown ID')
    if set(study['cohorts'])!={'pilot','expanded','smoke'}: raise ValueError('cohort names changed')
    return study

def prepare(source,output=ROOT):
    source=Path(source); output=Path(output)
    if (output/'data/study.json').exists() or (output/'evaluation/gold.json').exists(): raise FileExistsError('prepared output exists')
    manifest=read(source/'frozen/manifest.json')
    hashes={}
    for name in ('data/metadata-val.json','data/registry.json','data/val.jsonl','data/split_boundaries.json'):
        hashes[name]=sha(source/name)
        if hashes[name]!=manifest['files'][name] or (name in DATA_SHA and hashes[name]!=DATA_SHA[name]): raise ValueError('frozen input hash mismatch: '+name)
    rows=read(source/'data/metadata-val.json')
    registry_raw=read(source/'data/registry.json')['categories']
    registry={r['category']:{k:v for k,v in r.items() if k not in ('category','category_id')} for r in registry_raw}
    if len(rows)!=422 or len(registry)!=12 or any(r['category']!=r['category_id'] for r in registry_raw): raise ValueError('frozen dataset shape changed')
    cohorts={k.removesuffix('20').removesuffix('60').removesuffix('2'):v for k,v in manifest['cohorts'].items()}
    if {k:len(v) for k,v in cohorts.items()}!={'pilot':20,'expanded':60,'smoke':2}: raise ValueError('cohort sizes changed')
    by_id={r['id']:r for r in rows}
    val=[json.loads(line) for line in (source/'data/val.jsonl').read_text(encoding='utf-8').splitlines()]
    if len({r['source_id'] for r in val})!=len(val): raise ValueError('duplicate gold identity')
    selected={i for group in cohorts.values() for i in group}
    answers={}
    for r in val:
        metadata=by_id[r['source_id']]
        if metadata['domain']!=r['domain'] or metadata['metadata']['field_name']!=r['field_name']: raise ValueError('metadata/gold binding mismatch')
        if r['ground_truth'] not in registry or registry[r['ground_truth']]['grading_level']!=r['ground_truth_level']: raise ValueError('gold outside registry')
        if r['source_id'] in selected: answers[r['source_id']]={'category':r['ground_truth'],'grading_level':r['ground_truth_level']}
    if set(answers)!=selected: raise ValueError('missing selected gold')
    supported=sorted({answers[i]['category'] for i in cohorts['pilot']+cohorts['expanded']})
    if len(supported)!=8: raise ValueError('Val80 class identity changed')
    study=validate_study({'schema':1,'cohorts':cohorts,'rows':rows,'registry':registry,'budget':BUDGET,
                          'source_hashes':hashes,'source_manifest_sha256':sha(source/'frozen/manifest.json')})
    save(output/'data/study.json',study)
    save(output/'data/study-manifest.json',{'study_sha256':sha(output/'data/study.json')})
    save(output/'evaluation/gold.json',{'answers':answers,'supported_classes':supported,'source_hashes':hashes})
    return study

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--source',type=Path,required=True); p.add_argument('--output',type=Path,default=ROOT)
    a=p.parse_args(argv); result=prepare(a.source,a.output); print(json.dumps({'cohorts':{k:len(v) for k,v in result['cohorts'].items()},'rows':len(result['rows'])})); return 0
if __name__=='__main__': raise SystemExit(main())
