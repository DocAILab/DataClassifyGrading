"""Explicit no-gold allowlist and immutable evaluation package preparation."""
import argparse
import json
from pathlib import Path
import shutil
from io_utils import ROOT,read,put,sha
from gate import validate_ids

TRAIN=ROOT.parent/'finance_program_student_sft_20261016_r1'
TEACHER=ROOT.parent/'finance_program_teacher_balanced_20261016_r9'
OLD=ROOT.parent/'finance_program_f66_nontraining_20261014_r6'


def prepare():
    source=read(OLD/'data/study.json');sft=read(TEACHER/'study.json')
    assert source['registry']==sft['registry'] and source['budget']==sft['budget']
    assert not {r['id'] for r in source['rows']} & {r['id'] for r in sft['rows']}
    ids=source['cohorts']['pilot']+source['cohorts']['expanded'];smoke=source['cohorts']['smoke']
    validate_ids(ids,smoke,{r['id'] for r in sft['rows']})
    study={**source,'cohorts':{'smoke':smoke,'val80':ids}}
    put(ROOT/'data/study.json',study)
    trainrows=[json.loads(l) for l in (TRAIN/'data/planner200.jsonl').read_text(encoding='utf-8').splitlines()]
    systems={r['messages'][0]['content'] for r in trainrows};assert len(systems)==1
    put(ROOT/'data/protocol.json',{'planner_system':next(iter(systems)),
        'generation_eos_note':'both actual model EOS248044 and supervised tokenizer EOS248046; identical in both arms'})
    for folder in ('program_loop','vendor','tests','reviews'):
        source_dir=TEACHER/'runtime'/folder
        for p in source_dir.rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:
                target=ROOT/'runtime'/folder/p.relative_to(source_dir)
                target.parent.mkdir(parents=True,exist_ok=True)
                if target.exists():assert sha(target)==sha(p)
                else:shutil.copyfile(p,target)
    shutil.copyfile(TEACHER/'runtime/runtime_identity.py',ROOT/'runtime/runtime_identity.py')
    (ROOT/'legacy').mkdir(exist_ok=False)
    for n in ('local_smoke.py','download_model.py'):shutil.copyfile(OLD/n,ROOT/'legacy'/n)
    shutil.copyfile(TRAIN/'model-identity.json',ROOT/'model-identity.json')
    # Offline labels remain in the existing local-only gold file, never copied into package.
    put(ROOT/'reviews/selection-proof.json',{'val_ids':ids,'smoke_ids':smoke,'val_sources':80,'sft_sources':1000,
        'sft_val_id_intersection':0,'old_study_sha256':sha(OLD/'data/study.json'),
        'gold_uploaded':False,'same_registry_and_budget':True,'source_runtime_signature':read(TEACHER/'manifest.json')['runtime_signature']})
    return {'val80':len(ids),'smoke':len(smoke),'no_gold':True}


def freeze():
    allowed=[]
    for folder in ('runtime','legacy','data','tests','fixtures'):
        allowed.extend(p for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    allowed.extend(ROOT.glob('*.py'));allowed.append(ROOT/'model-identity.json')
    allowed=sorted(set(allowed))
    config={'training_root':'/home/zhubh/finance_program_grpo_full_migrated_20261027_r3/dependencies/'+TRAIN.name,
        'training_manifest_sha256':'de6591df902e08279e44bdb7d156c01f2b0e7421e3c9d267ae41c9fd5c0b3946',
        'training_data_sha256':'f93ffa2dd9ff8aae06c3872a1e4c0a6c532b7e733c485912ec0740e052cd0afa',
        'adapter':'/home/zhubh/finance_program_grpo_full_migrated_20261027_r3/dependencies/'+TRAIN.name+'/formal/final',
        'model':'/home/zhubh/finance_program_grpo_full_migrated_20261027_r3/dependencies/finance_program_5090_smoke_20261010/models/Qwen3.5-9B',
        'python':'/home/zhubh/finance_program_grpo_full_migrated_20261027_r3/dependencies/finance_program_5090_smoke_20261010/env/bin/python',
        'deps':'/home/zhubh/finance_program_grpo_full_migrated_20261027_r3/dependencies/finance_program_adapter_compare_20261012_r2/deps',
        'gpu_index':'0','gpu_uuid':'GPU-4e00b91b-5123-d78d-35f3-e72212f3f588',
        'shared_lock':'/home/zhubh/finance_program_grpo_full_migrated_20261027_r3/dependencies/finance_program_5090_smoke_20261010/gpu0-experiment.lock',
        'runtime_signature':'c34105f2767513d7d50a1c5fa15f8123e790865d25da681efa049182110132f8',
        'poll_seconds':30,'wait_hours':24,'stage_timeout_seconds':18000,'val80':80,'pair_count':160,
        'model_calls_no_api':True,'eos_ids':[248044,248046],'one_formal_attempt':True,
        'skill_backend':'base_no_adapter','planner_conditions':['base','sft'],'gold_local_only':True,
        'recovery_parent':'/home/zhubh/finance_program_grpo_full_migrated_20261027_r3/dependencies/finance_program_student_eval_20261016_r1',
        'recovery_parent_archive_sha256':'e4ed1253cf5b61d39b755091f8197bc9c7c2e801496820e33268b085354536e6',
        'cpu_validator_version':2,'generation_strategy_unchanged':True}
    put(ROOT/'manifest.json',{'schema':1,'config':config,'files':[{'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size} for p in allowed]})
    return {'manifest_sha256':sha(ROOT/'manifest.json'),'files':len(allowed)}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','freeze'));a=p.parse_args()
    print(json.dumps(prepare() if a.mode=='prepare' else freeze()))
