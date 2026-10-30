"""CPU-only replacement audit for PEFT-normalized target suffix serialization.

The original frozen training verifier remains unchanged and its failed exit is preserved.
This audit retains all original gates and validates the actual forty expanded layers.
"""
from collections import Counter
from pathlib import Path
import argparse
import json
import os
import sys
from io_utils import ROOT,read,put,sha,verify
from gate import validate_lora_inventory


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--training-root',type=Path,required=True);args=parser.parse_args()
    root=args.training_root;m=verify();config=m['config']
    if sha(root/'manifest.json')!=config['training_manifest_sha256']:raise ValueError('wrong frozen training identity')
    training=read(root/'manifest.json');c=training['config']
    for item in training['files']:
        p=root/item['path']
        if not p.resolve().is_relative_to(root.resolve()) or sha(p)!=item['sha256']:raise ValueError('training frozen file differs')
    if sha(root/'data/planner200.jsonl')!=config['training_data_sha256']:raise ValueError('training data identity differs')
    for name in ('trial','formal','pipeline','archive'):
        if read(root/'runs'/(name+'.exit.json'))['exit_code']!=0:raise ValueError('not all training exits succeeded')
    sys.path.insert(0,str(root))
    from training_contract import validate_trial
    validate_trial(read(root/'trial/proof.json'),c)
    complete=read(root/'formal/completion.json')
    if (complete['global_step']!=13 or complete['sources']!=200 or complete['epoch']!=1.
            or complete['manifest_sha256']!=sha(root/'manifest.json')):raise ValueError('incomplete/wrong training')
    rows=[json.loads(s) for s in (root/'data/planner200.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
    accounting=read(root/'formal/source-accounting.json');audit=read(root/'reviews/token-audit-01.json')
    if (len(rows)!=200 or len({r['source_id'] for r in rows})!=200 or len(audit['per_source'])!=200
            or Counter(accounting['consumed_ids'])!=Counter(r['source_id'] for r in rows)
            or accounting['finite_gradient_steps']!=list(range(1,14))):raise ValueError('full source accounting required')
    from transformers import AutoTokenizer,AutoConfig,AutoModelForImageTextToText
    tok=AutoTokenizer.from_pretrained(c['model'],local_files_only=True)
    for row,stat in zip(rows,audit['per_source']):
        prompt=tok.apply_chat_template(row['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        prefix=tok(prompt,add_special_tokens=False)['input_ids']
        full=tok(prompt+row['response']+tok.eos_token,add_special_tokens=False)['input_ids'];target=full[len(prefix):]
        if (row['source_id']!=stat['source_id'] or full[:len(prefix)]!=prefix or len(full)!=stat['length']
                or len(target)!=stat['supervised_tokens'] or len(full)>c['max_length_limit']
                or tok.decode(target,skip_special_tokens=False)!=row['response']+tok.eos_token
                or full[-1]!=tok.eos_token_id):raise ValueError('independent full token/mask/EOS audit failed')
    import torch
    from accelerate import init_empty_weights
    from safetensors.torch import load_file
    final=root/'formal/final';adapter=final/'adapter_model.safetensors'
    if sha(adapter)!=complete['adapter_sha256'] or sha(final/'adapter_config.json')!=complete['adapter_config_sha256']:
        raise ValueError('final adapter/config bytes changed')
    tensors=load_file(str(adapter),device='cpu')
    if (not tensors or not all(torch.isfinite(t).all().item() for t in tensors.values())
            or not any('lora_B' in n and torch.count_nonzero(t).item() for n,t in tensors.items())):
        raise ValueError('adapter nonfinite/empty/not updated')
    baseconfig=AutoConfig.from_pretrained(c['model'],local_files_only=True)
    with init_empty_weights():model=AutoModelForImageTextToText.from_config(baseconfig,dtype=torch.float16,attn_implementation='sdpa')
    if not all(p.device.type=='meta' for p in model.parameters()) or torch.cuda.is_initialized():
        raise RuntimeError('architecture audit must remain meta-only without GPU initialization')
    shapes={n:[layer.out_features,layer.in_features] for n,layer in model.named_modules() if isinstance(layer,torch.nn.Linear)}
    inventory=[{'name':n,'shape':list(t.shape)} for n,t in tensors.items()]
    lora_config=read(final/'adapter_config.json')
    if lora_config['base_model_name_or_path']!=c['model']:raise ValueError('wrong adapter base model')
    validation=validate_lora_inventory(lora_config,inventory,read(root/'formal/trainable-parameters.json')['targets'],shapes)
    for item in read(root/'model-identity.json')['files']:
        if sha(Path(c['model'])/item['path'])!=item['sha256']:raise ValueError('original base files changed')
    result={'independent_cpu_audit_passed':True,'all_200_full_token_masks_exact':True,
        'global_step':13,'unique_sources':200,'epoch':1.,'adapter_tensors':len(tensors),
        'adapter_sha256':sha(adapter),'adapter_config_sha256':sha(final/'adapter_config.json'),
        'adapter_language_only_finite_nonzero':True,'base_all_files_sha_unchanged':True,
        'manifest_sha256':sha(root/'manifest.json'),'no_gpu_model_or_eval_launched':True,
        'cpu_validator_version':2,'verifier_sha256':sha(ROOT/'verify_training.py'),
        'validator_manifest_sha256':sha(ROOT/'manifest.json'),'target_inventory_validation':validation,
        'original_failed_verifier_preserved':True,'architecture_meta_only':True}
    put(root/'reviews/post-training-cpu-proof.json',result)
    print(json.dumps(result))


if __name__=='__main__':main()
