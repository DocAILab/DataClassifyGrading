"""Fresh-base Planner LoRA: actual longest-source trial, then one full epoch."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

from runtime_io import digest, read, put, verify_manifest
from token_audit import encode_training_row
from training_contract import expected_steps, validate_trial


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('trial', 'formal'), required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    manifest = verify_manifest(root)
    c = manifest['config']
    import torch
    import peft
    import transformers
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoTokenizer, Trainer, TrainingArguments, TrainerCallback, set_seed
    from completion_loss import completion_cross_entropy
    if (not torch.cuda.is_available() or torch.cuda.device_count() != 3
            or not torch.cuda.is_bf16_supported()):
        raise RuntimeError('three BF16 CUDA GPUs required')
    output = root / args.mode
    output.mkdir(exist_ok=False)
    started = time.time()
    if args.mode == 'formal':
        trial = read(root / 'trial/proof.json')
        validate_trial(trial, c)
        if trial['manifest_sha256'] != digest(root / 'manifest.json'):
            raise ValueError('trial belongs to a different frozen run')
    set_seed(c['seed'])
    tok = AutoTokenizer.from_pretrained(c['model'], local_files_only=True)
    rows = [json.loads(line) for line in (root / 'data/planner200.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    if len(rows) != 200 or len({r['source_id'] for r in rows}) != 200:
        raise ValueError('exact frozen 200 unique sources required')
    enc = [encode_training_row(r, tok, c['max_length_limit']) for r in rows]
    audit = read(root / 'reviews/token-audit-01.json')
    for row, item, stat in zip(rows, enc, audit['per_source']):
        count = sum(x != -100 for x in item['labels'])
        if (stat['source_id'] != row['source_id'] or stat['length'] != len(item['input_ids'])
                or stat['supervised_tokens'] != count
                or tok.decode(item['input_ids'][-count:], skip_special_tokens=False) != row['response'] + tok.eos_token):
            raise ValueError('full token/mask audit changed')
    if max(len(x['input_ids']) for x in enc) != c['max_length']:
        raise ValueError('actual longest length changed')
    model = AutoModelForImageTextToText.from_pretrained(c['model'], local_files_only=True,
        dtype=torch.bfloat16, attn_implementation='sdpa', device_map='balanced',
        max_memory={0: '27GiB', 1: '27GiB', 2: '27GiB'})
    placements = {str(d) for d in model.hf_device_map.values()}
    if any(not placements.intersection({str(i), 'cuda:' + str(i)}) for i in range(3)):
        raise RuntimeError('base must be sharded over all three GPUs')
    model.config.use_cache = False
    targets = [name for name, layer in model.named_modules()
        if name.startswith('model.language_model.layers.')
        and name.endswith(('.q_proj', '.v_proj', '.in_proj_qkv'))
        and isinstance(layer, torch.nn.Linear)]
    if len(targets) != 40:
        raise ValueError('expected 40 language LoRA targets')
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=.05,
        target_modules=targets, bias='none', task_type='CAUSAL_LM'))
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]

    def frozen():
        return bool(trainable) and all('.lora_' in n and '.language_model.layers.' in n
            for n, p in model.named_parameters() if p.requires_grad)

    if not frozen() or sorted({str(p.device) for _, p in trainable}) != ['cuda:0', 'cuda:1', 'cuda:2']:
        raise ValueError('only distributed language LoRA may be trainable; base/vision frozen')
    put(output / 'trainable-parameters.json', {'count': sum(p.numel() for _, p in trainable),
        'targets': targets, 'devices': sorted({str(p.device) for _, p in trainable}), 'base_and_vision_frozen': frozen()})
    for i in range(3):
        torch.cuda.reset_peak_memory_stats(i)

    def collate(items):
        if len(items) != 1:
            raise ValueError('micro batch must be one')
        item = items[0]
        values = {k: torch.tensor(item[k], dtype=torch.long).unsqueeze(0)
            for k in ('input_ids', 'attention_mask', 'labels')}
        if '_source_id' in item:
            values['_source_id'] = item['_source_id']
        return values

    def loss_for(inputs):
        labels = inputs.pop('labels')
        count = int(labels.ne(-100).sum())
        with torch.autograd.graph.save_on_cpu(pin_memory=True):
            result = model(**inputs, logits_to_keep=count + 1)
            loss = completion_cross_entropy(result.logits, labels.to(result.logits.device))
        if not torch.isfinite(loss.detach()).item():
            raise FloatingPointError('nonfinite training loss')
        return loss, result

    if args.mode == 'trial':
        idx = next(i for i, r in enumerate(rows) if r['source_id'] == c['longest_source_id'])
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.enable_input_require_grads()
        model.train()
        device = model.get_input_embeddings().weight.device
        inputs = {k: v.to(device) for k, v in collate([enc[idx]]).items()}
        optimizer = torch.optim.AdamW([p for _, p in trainable], lr=1e-4, weight_decay=.01)
        loss, _ = loss_for(inputs)
        loss.backward()
        grads = [p.grad for _, p in trainable if p.grad is not None]
        finite = bool(grads) and all(torch.isfinite(g).all().item() for g in grads)
        if not finite:
            raise FloatingPointError('trial gradients missing/nonfinite')
        norm = torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1.)
        optimizer.step()
        updated = any('.lora_B.' in n and torch.count_nonzero(p.detach()).item() > 0 for n, p in trainable)
        proof = {'data_sha256': c['data_sha256'], 'manifest_sha256': digest(root / 'manifest.json'),
            'source_id': rows[idx]['source_id'], 'tokens': len(enc[idx]['input_ids']), 'gpu_uuids': c['gpu_uuids'],
            'target_modules': len(targets), 'base_and_vision_frozen': frozen(), 'lora_updated': updated,
            'gradients_finite': finite, 'loss_finite': True, 'loss': float(loss.detach()), 'grad_norm': float(norm),
            'trainable_devices': sorted({str(p.device) for _, p in trainable}),
            'peak_memory_allocated': {str(i): torch.cuda.max_memory_allocated(i) for i in range(3)},
            'wall_seconds': time.time() - started, 'formal_reloads_fresh_base': True}
        validate_trial(proof, c)
        if not all(torch.isfinite(p.detach()).all().item() for _, p in trainable):
            raise FloatingPointError('trial updated parameters nonfinite')
        put(output / 'proof.json', proof)
        print(json.dumps({'stage': 'trial_complete', **proof}), flush=True)
        return

    consumed = []
    gradient_steps = []

    class FiniteGradients(TrainerCallback):
        def on_pre_optimizer_step(self, args, state, control, **kwargs):
            grads = [p.grad for _, p in trainable if p.grad is not None]
            if not grads or not all(torch.isfinite(g).all().item() for g in grads):
                raise FloatingPointError('formal gradients missing/nonfinite')
            gradient_steps.append(state.global_step + 1)

    class PlannerTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            source = inputs.pop('_source_id')
            consumed.append(source)
            loss, result = loss_for(inputs)
            return (loss, result) if return_outputs else loss

    for row, item in zip(rows, enc):
        item['_source_id'] = row['source_id']
    training_args = TrainingArguments(output_dir=str(output), num_train_epochs=1,
        per_device_train_batch_size=1, gradient_accumulation_steps=16,
        learning_rate=1e-4, lr_scheduler_type='cosine', warmup_steps=3,
        weight_decay=.01, max_grad_norm=1., bf16=True, optim='adamw_torch',
        gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
        eval_strategy='no', save_strategy='steps', save_steps=8, logging_steps=1,
        report_to='none', push_to_hub=False, remove_unused_columns=False,
        dataloader_num_workers=0, dataloader_drop_last=False, seed=c['seed'], data_seed=c['seed'], disable_tqdm=True)
    trainer = PlannerTrainer(model=model, args=training_args, train_dataset=enc,
        data_collator=collate, processing_class=tok, callbacks=[FiniteGradients()])
    # Explicit example-average accumulation: custom loss does not consume num_items_in_batch.
    trainer.model_accepts_loss_kwargs = False
    result = trainer.train()
    wanted = expected_steps(len(rows), 16)
    if (trainer.state.global_step != wanted or trainer.state.epoch != 1.0
            or Counter(consumed) != Counter(r['source_id'] for r in rows)
            or gradient_steps != list(range(1, wanted + 1))):
        raise RuntimeError('incomplete epoch/source accounting/finite gradient updates')
    if not frozen() or not all(torch.isfinite(p.detach()).all().item() for _, p in trainable):
        raise FloatingPointError('invalid final trainable parameters')
    if not any('.lora_B.' in n and torch.count_nonzero(p.detach()).item() > 0 for n, p in trainable):
        raise RuntimeError('no nonzero LoRA update')
    final = output / 'final'
    trainer.save_model(str(final))
    tok.save_pretrained(final)
    trainer.save_state()
    if not (final / 'adapter_model.safetensors').is_file() or list(final.glob('model*.safetensors')):
        raise RuntimeError('adapter-only checkpoint required')
    put(output / 'source-accounting.json', {'consumed_ids': consumed, 'unique_sources': len(set(consumed)),
        'finite_gradient_steps': gradient_steps, 'last_partial_accumulation_retained': True})
    put(output / 'completion.json', {'global_step': trainer.state.global_step, 'epoch': trainer.state.epoch,
        'sources': len(consumed), 'metrics': result.metrics, 'base_and_vision_frozen': frozen(),
        'adapter_finite_and_updated': True, 'adapter_sha256': digest(final / 'adapter_model.safetensors'),
        'adapter_config_sha256': digest(final / 'adapter_config.json'), 'manifest_sha256': digest(root / 'manifest.json'),
        'trial_proof_sha256': digest(root / 'trial/proof.json'), 'torch': torch.__version__,
        'transformers': transformers.__version__, 'peft': peft.__version__,
        'wall_seconds': time.time() - started, 'evaluation_performed': False})
    print('STUDENT_PLANNER_SFT_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
