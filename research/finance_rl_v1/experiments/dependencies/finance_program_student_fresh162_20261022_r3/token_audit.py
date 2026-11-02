"""Actual pinned local CPU tokenizer; no model/CUDA and no truncation."""
import json
from pathlib import Path
from contract import ROOT,MAX_LENGTH,load_train,put,sha
from deps import identity,encoding
BASE='/home/zhubh/finance_program_grpo_full_migrated_20261027_r3/dependencies/finance_program_5090_smoke_20261010/models/Qwen3.5-9B'
def tokenizer():
    identity.verify_tokenizer_identity(BASE,ROOT/'model-identity.json')
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(BASE,local_files_only=True,use_fast=True)
    if not tok.is_fast:raise ValueError('original fast tokenizer required')
    return tok
def main():
    rows=load_train();tok=tokenizer();stats=[]
    for row in rows:
        try:
            e=encoding.encode_completion(tok,row,MAX_LENGTH)
            stats.append(dict(id=row['id'],passed=True,total_tokens=len(e['input_ids']),prompt_tokens=e['prompt_length'],target_tokens=e['target_length'],truncated=False))
        except ValueError as exc:stats.append(dict(id=row['id'],passed=False,error=str(exc),truncated=False))
    result=dict(rows=162,train_sha256=sha(ROOT/'data/train162.jsonl'),per_example=stats,all_fit=all(s['passed'] for s in stats),max_length=MAX_LENGTH,
        max_total_tokens=max((s.get('total_tokens',0) for s in stats),default=0),model_calls=0,old_adapters_loaded=False)
    put(ROOT/'reviews/tokenizer-audit-01.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='per_example'}))
    if not result['all_fit']:raise SystemExit('original162 do not all fit; no silent truncation/GPU')
if __name__=='__main__':main()
