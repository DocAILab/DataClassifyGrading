"""Full, untruncated student chat encoding and assistant-only suffix audit."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def encode_training_row(row,tokenizer,max_length):
    prompt=tokenizer.apply_chat_template(row['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    if not tokenizer.eos_token:raise ValueError('tokenizer requires EOS')
    prefix=tokenizer(prompt,add_special_tokens=False)['input_ids']
    full=tokenizer(prompt+row['response']+tokenizer.eos_token,add_special_tokens=False)['input_ids']
    if full[:len(prefix)]!=prefix:raise ValueError('prompt is not an exact token prefix')
    if len(full)>max_length:raise ValueError('context too long; no truncation permitted')
    if len(full)<=len(prefix):raise ValueError('empty assistant suffix')
    return {'input_ids':full,'attention_mask':[1]*len(full),'labels':[-100]*len(prefix)+full[len(prefix):]}


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--model',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--limit',type=int,default=262144);a=p.parse_args()
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    if not tok.is_fast:raise ValueError('fast tokenizer required')
    rows=[json.loads(x) for x in a.data.read_text(encoding='utf-8').splitlines() if x.strip()]
    if len(rows)!=200 or len({r['source_id'] for r in rows})!=200:raise ValueError('exactly 200 sources required')
    stats=[]
    for row in rows:
        enc=encode_training_row(row,tok,a.limit);count=sum(x!=-100 for x in enc['labels']);boundary=len(enc['labels'])-count
        assert all(x==-100 for x in enc['labels'][:boundary]) and enc['labels'][boundary:]==enc['input_ids'][boundary:]
        assert tok.decode(enc['input_ids'][boundary:],skip_special_tokens=False)==row['response']+tok.eos_token
        assert enc['input_ids'][-1]==tok.eos_token_id
        stats.append({'source_id':row['source_id'],'length':len(enc['input_ids']),'prompt_tokens':boundary,'supervised_tokens':count})
    lengths=sorted(x['length'] for x in stats);report={'rows':len(rows),'data_sha256':hashlib.sha256(a.data.read_bytes()).hexdigest(),
        'min_length':min(lengths),'max_length':max(lengths),'p95_length':lengths[int(.95*(len(lengths)-1))],
        'max_length_limit':math.ceil(max(lengths)/1024)*1024,'total_tokens':sum(lengths),
        'min_supervised_tokens':min(x['supervised_tokens'] for x in stats),'max_supervised_tokens':max(x['supervised_tokens'] for x in stats),
        'longest_source_id':max(stats,key=lambda x:x['length'])['source_id'],'truncated_rows':0,'all_prompt_positions_masked':True,
        'all_assistant_targets_exact_with_eos':True,'enable_thinking':False,'model_tokenizer_sha256':hashlib.sha256((a.model/'tokenizer.json').read_bytes()).hexdigest(),
        'tokenizer_config_sha256':hashlib.sha256((a.model/'tokenizer_config.json').read_bytes()).hexdigest(),'per_source':stats}
    with a.output.open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print(json.dumps({k:v for k,v in report.items() if k!='per_source'},ensure_ascii=False))


if __name__=='__main__':main()
