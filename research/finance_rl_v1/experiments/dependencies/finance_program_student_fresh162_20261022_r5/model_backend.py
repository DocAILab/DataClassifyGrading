"""One base and only the new final adapter; base-only versus new Planner."""
from contextlib import nullcontext
import json
from pathlib import Path
import time
from contract import ROOT,put,sha,disable_for,validate_training_counts
from deps import identity
from train import load_model,tensor_sha
from token_audit import BASE,tokenizer
from eval_runtime import raw_bytes,require
sha256_bytes=identity.sha256_bytes
exclusive_json=put
def generation_options(admission):
    identity.check_deadline(admission)
    return dict(do_sample=False,max_new_tokens=512,max_time=min(90.,admission['gpu_deadline_unix']-time.time()),eos_token_id=[248044,248046],use_cache=True)

class ModelBackend:
    def __init__(self,admission):
        self.admission=admission;self.runtime=None;self.current_arm='N';self.load_failure=None
        self.last_generation_metadata={};self.last_call_evidence={}
        child=ROOT/'runs/training/final-adapter';complete=json.loads((child.parent/'completion.json').read_bytes())
        validate_training_counts(complete)
        self.expected={k:sha(child/k) for k in ('adapter_model.safetensors','adapter_config.json')}
        if self.expected!=complete['final_adapter']['files_sha256']:raise ValueError('new final adapter byte identity differs')
        if complete['admission']['gpu_deadline_unix']!=admission['gpu_deadline_unix']:raise ValueError('train/eval clock must be shared')
        self.identity=dict(base=identity.verify_base_identity(BASE,ROOT/'model-identity.json'),
            tokenizer=identity.verify_tokenizer_identity(BASE,ROOT/'model-identity.json'),new_adapter=self.expected,
            starting_point='original_base_fresh_lora',old_adapter_loaded=False,train_sha256=complete['train_sha256'])
        self.expected_tensors=complete['final_tensors_sha256']
    def _ensure_loaded(self):
        if self.runtime is not None:return
        if self.load_failure is not None:raise ValueError('model load is not retried: '+self.load_failure)
        self.runtime=load_model(self.admission,child=ROOT/'runs/training/final-adapter')
        self.runtime['tokenizer']=tokenizer();model=self.runtime['model'];torch=self.runtime['torch']
        self.named=self.runtime['named'];self.modules={n:m for n,m in model.named_modules() if n in identity.ORIGINAL_LORA_MODULES}
        if set(self.modules)!=identity.ORIGINAL_LORA_MODULES:raise ValueError('all40 route modules required')
        self.weight_shas={n:tensor_sha(p) for n,p in self.named}
        if self.weight_shas!=self.expected_tensors:raise ValueError('loaded new final162 tensors differ')
        for _,p in model.named_parameters():p.requires_grad_(False);p.grad=None
        model.eval();model.config.use_cache=True
    def _select(self,arm):
        if arm not in ('F','N'):raise ValueError('only base(F) and fresh162(N) arms permitted')
        self.current_arm=arm
    def _state(self,disabled):
        model=self.runtime['model'];torch=self.runtime['torch'];routes={}
        for n,m in self.modules.items():
            if bool(m.disable_adapters)!=disabled or list(m.active_adapters)!=['default']:raise ValueError('actual adapter routing differs')
            routes[n]=dict(disabled=bool(m.disable_adapters),active_adapters=list(m.active_adapters))
        for n,p in model.named_parameters():
            if p.requires_grad or p.grad is not None or p.dtype!=(torch.float32 if n in identity.ORIGINAL_LORA_NAMES else torch.bfloat16):raise ValueError('eval frozen dtype route contract')
        hashes={n:tensor_sha(p) for n,p in self.named}
        if hashes!=self.weight_shas or model.training:raise ValueError('evaluation adapter weights changed')
        return dict(arm=self.current_arm,effective_route='base_only' if disabled else 'N',modules=routes,tensors_sha256=hashes,training=False,all_parameters_frozen=True)
    def generate(self, messages, arm, stage, destination):
        from paired_eval import BudgetExhausted
        destination = Path(destination)
        self.last_generation_metadata = {}
        self.last_call_evidence = {}
        record = dict(arm=arm, stage=stage, messages=messages, messages_sha256=sha256_bytes(raw_bytes(messages)),
                      model_identity=self.identity, generation_invoked=False, generation_returned=False,
                      raw_text=None, output_token_ids=None, error=None)
        started = time.monotonic()
        exclusive_json(destination / 'model-request.json', record)
        try:
            self._ensure_loaded()
            self._select(arm)
            torch, model, tokenizer = self.runtime['torch'], self.runtime['model'], self.runtime['tokenizer']
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            ids = tokenizer(prompt, add_special_tokens=False)['input_ids']
            require(ids and all(type(value) is int for value in ids), 'nonempty original integer prompt tokens required')
            require(len(ids) + 512 <= 49152, 'context exceeds 49152 with full 512 generation allowance; no truncation')
            record.update(prompt=prompt, prompt_token_ids=ids, prompt_sha256=sha256_bytes(prompt.encode('utf-8')),
                          prompt_token_ids_sha256=sha256_bytes(raw_bytes(ids)))
            device = model.get_input_embeddings().weight.device
            input_ids = torch.tensor([ids], dtype=torch.long, device=device)
            attention = torch.ones_like(input_ids)
            outer = self._state(False)
            record['route_before'] = outer
            context = model.disable_adapter() if disable_for(arm,stage) else nullcontext()
            try:
                with context:
                    record['route_enter'] = self._state(disable_for(arm,stage))
                    options = generation_options(self.admission)
                    record['generation_options'] = options
                    record['generation_started'] = True
                    exclusive_json(destination / 'generation-started.json', record)
                    call_start = time.monotonic()
                    autocast = torch.autocast(device_type='cuda', dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext()
                    with torch.inference_mode(), autocast:
                        record['generation_invoked'] = True
                        output = model.generate(input_ids=input_ids, attention_mask=attention, **options)
                    record['generation_returned'] = True
                    generated = output[0, len(ids):].detach().cpu().tolist()
                    text = tokenizer.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    elapsed = max(0., time.monotonic() - call_start)
                    eos = bool(generated and generated[-1] in (248044, 248046))
                    reason = 'eos' if eos else ('max_new_tokens' if len(generated) >= 512 else 'max_time_or_early_stop')
                    record.update(output_token_ids=generated, raw_text=text, output_text_sha256=sha256_bytes(text.encode('utf-8')),
                                  output_token_ids_sha256=sha256_bytes(raw_bytes(generated)), termination_reason=reason,
                                  route_after_forward=self._state(disable_for(arm,stage)))
                    self.last_generation_metadata = dict(input_tokens=len(ids), output_tokens=len(generated), error=None,
                                                         truncation_at_budget=not eos)
                    if elapsed > options['max_time'] or not eos:
                        raise RuntimeError('generation did not finish with an actual EOS within the admitted call budget')
                    return text
            finally:
                # PEFT re-enabling an adapter may mark its parameters trainable.
                # Restore evaluation's full freeze after either success or error.
                for _, parameter in model.named_parameters():
                    parameter.requires_grad_(False)
                    parameter.grad = None
                record['route_exit'] = self._state(False)
        except Exception as exc:
            if self.runtime is None:
                self.load_failure = str(exc)
            record['error'] = str(exc)
            record['error_type'] = type(exc).__name__
            self.last_generation_metadata = dict(self.last_generation_metadata, error=str(exc))
            if time.time() >= self.admission['gpu_deadline_unix']:
                raise BudgetExhausted('original absolute GPU deadline exhausted') from exc
            raise
        finally:
            record['elapsed_seconds'] = max(0., time.monotonic() - started)
            record['generation'] = self.last_generation_metadata
            self.last_call_evidence = {k: record[k] for k in ('generation_invoked', 'generation_returned')}
            exclusive_json(destination / 'model-returned.json', record)
