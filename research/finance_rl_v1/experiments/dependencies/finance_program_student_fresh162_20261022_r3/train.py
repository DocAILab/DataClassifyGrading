"""Authorized fresh162 rerun; atomic checkpoints at every optimizer boundary."""
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import time
from types import SimpleNamespace
from uuid import UUID
from contract import ROOT,SEED,MAX_LENGTH,load_train,batches,divisor,fresh_lora_settings,put,sha
from deps import identity,encoding,math_runtime,configure_gradient_checkpointing
from token_audit import BASE,tokenizer
from runtime_safety import guard_resources,persist_boundary

def tensor_sha(tensor):
    import hashlib
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

def configure_devices(admission,torch):
    identity.validate_admission(admission)
    uuids=admission['gpu_uuids']
    if len(uuids)!=3 or os.environ.get('CUDA_VISIBLE_DEVICES')!=','.join(uuids) or torch.cuda.device_count()!=3:
        raise ValueError('exact three admitted visible GPUs required')
    for i,u in enumerate(uuids):
        props=torch.cuda.get_device_properties(i)
        with torch.cuda.device(i):bf16=torch.cuda.is_bf16_supported()
        if 'GPU-'+str(UUID(str(props.uuid).removeprefix('GPU-')))!=u or not bf16:
            raise ValueError('actual UUID/BF16 differs')
        torch.cuda.set_per_process_memory_fraction(admission['allocator_limit_bytes'][i]/props.total_memory,i)

def load_model(admission, *, child=None):
    import torch,transformers,peft
    configure_devices(admission,torch)
    guard_resources(ROOT,torch,preload=True)
    transformers.set_seed(SEED)
    base=transformers.AutoModelForImageTextToText.from_pretrained(BASE,local_files_only=True,dtype=torch.bfloat16,
        attn_implementation='sdpa',device_map='balanced',max_memory=dict(enumerate(admission['model_max_memory_bytes'])))
    if {str(d).removeprefix('cuda:') for d in base.hf_device_map.values()}!={'0','1','2'}:
        raise ValueError('base must occupy all admitted GPUs without CPU/disk')
    if child is None:
        targets=[n for n,m in base.named_modules() if n.startswith('model.language_model.layers.')
            and n.endswith(('.q_proj','.v_proj','.in_proj_qkv')) and isinstance(m,torch.nn.Linear)]
        model=peft.get_peft_model(base,peft.LoraConfig(**fresh_lora_settings(targets)))
    else:
        # Only this run's final new adapter, never A/B80.
        child=Path(child).resolve(strict=True)
        if child!=ROOT/'runs/training/final-adapter':raise ValueError('only unique new final adapter may load for evaluation')
        model=peft.PeftModel.from_pretrained(base,str(child),is_trainable=True,local_files_only=True)
    model.config.use_cache=False
    named=identity.validate_parameters(model,torch,cast=True)
    if child is None:
        if not all(torch.count_nonzero(p).item()==0 for n,p in named if '.lora_B.' in n):
            raise ValueError('fresh LoRA initialization B must be zero')
        if not all(torch.count_nonzero(p).item()>0 for n,p in named if '.lora_A.' in n):
            raise ValueError('fresh LoRA A must contain actual random initialization')
    return dict(torch=torch,transformers=transformers,peft=peft,model=model,named=named)

def run():
    from supervisor import verify
    verify()
    admission=json.loads((ROOT/'admission.json').read_bytes())
    identity.check_deadline(admission)
    run=ROOT/'runs/training';run.mkdir(parents=True,exist_ok=False)
    put(run/'started.json',dict(pid=os.getpid(),time_unix=time.time(),starting_point='original_base_fresh_lora',old_adapter_loaded=False))
    try:
        rows=load_train();tok=tokenizer()
        encoded=[SimpleNamespace(**encoding.encode_completion(tok,r,MAX_LENGTH)) for r in rows]
        proof=json.loads((ROOT/'reviews/tokenizer-audit-01.json').read_bytes())
        if not proof['all_fit'] or proof['train_sha256']!=sha(ROOT/'data/train162.jsonl') or len(proof['per_example'])!=162:
            raise ValueError('actual original CPU token proof required')
        if [len(e.input_ids) for e in encoded]!=[s['total_tokens'] for s in proof['per_example']]:raise ValueError('token audit changed')
        base_identity=identity.verify_base_identity(BASE,ROOT/'model-identity.json')
        put(run/'preload-identity.json',dict(base=base_identity,train_sha256=sha(ROOT/'data/train162.jsonl'),old200_targets=0,old_adapters_loaded=False))
        rt=load_model(admission);torch=rt['torch'];model=rt['model'];named=rt['named']
        initial={n:dict(sha256=tensor_sha(p),dtype=str(p.dtype),shape=list(p.shape),zero=bool(torch.count_nonzero(p).item()==0)) for n,p in named}
        put(run/'initial-lora.json',dict(new_initialization=True,old_adapter_loaded=False,tensors=initial,base_identity=base_identity))
        put(run/'checkpoint-configuration.json',configure_gradient_checkpointing(model))
        model.enable_input_require_grads();model.train()
        optimizer=torch.optim.AdamW([p for _,p in named],lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=.01)
        checkpoint_identity=dict(train_sha256=sha(ROOT/'data/train162.jsonl'),freeze_sha256=sha(ROOT/'freeze.json'),
            base=base_identity,seed=SEED,admission=admission,initial=initial)
        persist_boundary(run/'checkpoints',named,optimizer,step=0,consumed=[],identity=checkpoint_identity)
        consumed=[];steps=[]
        for step,indices in enumerate(batches(list(range(162))),1):
            identity.check_deadline(admission);optimizer.zero_grad(set_to_none=True)
            before={n:p.detach().cpu().clone() for n,p in named};losses=[]
            for i in indices:
                identity.check_deadline(admission)
                resources=guard_resources(ROOT,torch)
                event=dict(step=step,micro=i+1,id=rows[i]['id'],time_unix=time.time())
                event['resources']=resources
                put(run/f'micro-{i+1:03d}.forward-start.json',event)
                with torch.autograd.graph.save_on_cpu(pin_memory=True):
                    loss=math_runtime.completion_loss(model,encoded[i],torch)
                    value=float(loss.detach().item());losses.append(value)
                    put(run/f'micro-{i+1:03d}.forward-return.json',dict(event,loss=value,time_unix=time.time()))
                    (loss/divisor(indices)).backward()
                    put(run/f'micro-{i+1:03d}.backward-return.json',dict(event,time_unix=time.time()))
                del loss
                consumed.append(rows[i]['id'])
            norm,clipped=math_runtime.global_clip(named,torch)
            identity.check_deadline(admission)
            put(run/f'step-{step:02d}.optimizer-start.json',dict(step=step,time_unix=time.time()))
            optimizer.step()
            put(run/f'step-{step:02d}.optimizer-return.json',dict(step=step,time_unix=time.time()))
            for _,p in named:
                st=optimizer.state[p]
                if float(st['step'].item())!=step or any(v.dtype!=torch.float32 or not torch.isfinite(v).all().item() for v in st.values() if torch.is_tensor(v)):
                    raise ValueError('actual AdamW steps/state must be finite FP32')
            delta=math.sqrt(sum(float((p.detach().cpu()-before[n]).double().square().sum().item()) for n,p in named))
            if not math.isfinite(delta) or delta<=0:raise ValueError('actual update must be finite nonzero')
            identity.validate_parameters(model,torch)
            ev=dict(step=step,completed=True,micro_ids=consumed[-len(indices):],normalization_divisor=len(indices),loss=sum(losses)/len(losses),
                grad_norm=norm,clipped_grad_norm=clipped,update_norm=delta,memory=math_runtime.memory_snapshot(torch,model))
            put(run/f'step-{step:02d}.evidence.json',ev);steps.append(ev)
            persist_boundary(run/'checkpoints',named,optimizer,step=step,consumed=list(consumed),identity=checkpoint_identity)
            guard_resources(ROOT,torch)
        if consumed!=[r['id'] for r in rows] or len(steps)!=11:raise ValueError('whole162 once/11 actual updates required')
        identity.check_deadline(admission)
        final=math_runtime._save_final(model,run/'final-adapter',rt)
        if identity.verify_base_identity(BASE,ROOT/'model-identity.json')!=base_identity:raise ValueError('base changed')
        final_shas={n:tensor_sha(p) for n,p in named}
        if final_shas=={n:v['sha256'] for n,v in initial.items()}:raise ValueError('new adapter has no actual change')
        put(run/'completion.json',dict(rows=162,epochs=1,rows_consumed=consumed,optimizer_updates=11,optimizer_steps=steps,
            accumulation=16,learning_rate=1e-4,seed=SEED,starting_point='original_base_fresh_lora',old_adapters_loaded=False,
            train_sha256=sha(ROOT/'data/train162.jsonl'),final_adapter=final,final_tensors_sha256=final_shas,base_identity=base_identity,
            admission=admission,time_unix=time.time()))
        return 0
    except BaseException as exc:
        put(run/'failure.json',dict(error_type=type(exc).__name__,error=str(exc),time_unix=time.time(),retry_allowed=False))
        raise
if __name__=='__main__':raise SystemExit(run())
