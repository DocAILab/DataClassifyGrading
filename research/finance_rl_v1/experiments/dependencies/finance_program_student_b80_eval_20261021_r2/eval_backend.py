"""One BF16 base, exact FP32 A/child tensors, base-only Skill generation."""
from contextlib import nullcontext
import importlib
import importlib.util
import json
import math
from pathlib import Path
import time

from training_contract import (ORIGINAL_LORA_MODULES, ORIGINAL_LORA_NAMES, MANIFEST_SHA256, TRAIN_SHA256,
    ProtocolError, check_deadline, file_sha256, sha256_bytes, validate_admission, validate_parameters,
    verify_adapter_identity, verify_base_identity, verify_loaded_adapter, verify_tokenizer_identity)
from prepare_runtime import exclusive_json, raw_bytes, require


def generation_options(admission, now=None):
    now = time.time() if now is None else now
    check_deadline(admission, now)
    remaining = admission['gpu_deadline_unix'] - now
    return dict(do_sample=False, max_new_tokens=512, max_time=min(90., remaining),
                eos_token_id=[248044, 248046], use_cache=True)


def tensor_sha(tensor):
    return sha256_bytes(tensor.detach().cpu().contiguous().numpy().tobytes())


def normalized_adapter_config(config):
    # PEFT LoraConfig turns a list of target names into a set. Its saved list
    # order is not a parameterization change; regex strings remain exact.
    result = {k: v for k, v in config.items()
              if k not in {'inference_mode', 'base_model_name_or_path', 'auto_mapping'}}
    targets = result.get('target_modules')
    if isinstance(targets, list):
        require(all(isinstance(v, str) for v in targets) and len(set(targets)) == len(targets),
                'target module names must be unique strings')
        result['target_modules'] = sorted(targets)
    return result


class ModelBackend:
    def __init__(self, root, admission):
        self.root, self.admission = Path(root), admission
        validate_admission(admission)
        self.training_root = Path(admission.get('training_root', root)).resolve(strict=True)
        self.child = Path(admission.get('child_adapter_dir', self.training_root / 'runs/training/final-adapter')).resolve(strict=True)
        require(self.child == self.training_root / 'runs/training/final-adapter',
                'child must be the one final adapter from the admitted training root')
        self.original = Path(admission['original_adapter_dir']).resolve(strict=True)
        require(all(not (self.child / name).is_symlink() and (self.child / name).is_file()
                    for name in ('adapter_model.safetensors', 'adapter_config.json')),
                'child files must be physical regular files')
        self.identity = dict(base=verify_base_identity(admission['base_model_dir'], self.root / 'model-identity.json'),
                             original_A=verify_adapter_identity(self.original),
                             tokenizer=verify_tokenizer_identity(admission['base_model_dir'], self.root / 'model-identity.json'))
        self.identity['child'] = {name: file_sha256(self.child / name)
                                  for name in ('adapter_model.safetensors', 'adapter_config.json')}
        complete = json.loads((self.child.parent / 'completion.json').read_bytes())
        require(complete['cpu_test_only'] is False and complete['train_sha256'] == TRAIN_SHA256 and
                complete['manifest_sha256'] == MANIFEST_SHA256 and complete['epochs'] == 1 and
                len(complete['rows_consumed']) == len(set(complete['rows_consumed'])) == 80 and
                complete['rows'] == 80 and len(complete['optimizer_steps']) == 10 and
                [s['step'] for s in complete['optimizer_steps']] == list(range(1, 11)) and
                all(s['completed'] and math.isfinite(s['update_norm']) and s['update_norm'] > 0
                    for s in complete['optimizer_steps']) and
                complete['final_adapter']['files_sha256'] == self.identity['child'],
                'child must bind completed unique 80-row / 10-step training evidence')
        require(complete['admission']['gpu_deadline_unix'] == admission['gpu_deadline_unix'],
                'evaluation must inherit original training absolute GPU deadline')
        original_config = json.loads((self.original / 'adapter_config.json').read_bytes())
        child_config = json.loads((self.child / 'adapter_config.json').read_bytes())
        # These PEFT save metadata fields do not change the already attached adapter.
        require(normalized_adapter_config(original_config) == normalized_adapter_config(child_config),
                'A/child parameterized adapter configuration differs')
        loader_path = self.training_root / 'train_student.py'
        loader_sha = admission.get('runtime_loader_sha256')
        if self.training_root != self.root.resolve():
            require(isinstance(loader_sha, str) and file_sha256(loader_path) == loader_sha,
                    'corrected external training loader requires explicit pinned SHA')
        elif loader_sha is not None:
            require(file_sha256(loader_path) == loader_sha, 'runtime loader SHA mismatch')
        self.identity.update(runtime_loader_sha256=file_sha256(loader_path),
                             child_adapter_dir=str(self.child), training_root=str(self.training_root))
        self.loader_path = loader_path
        self.runtime = None
        self.current_arm = None
        self.last_generation_metadata = {}
        self.last_call_evidence = {}
        self.load_failure = None

    def _ensure_loaded(self):
        if self.runtime is not None:
            return
        if self.load_failure is not None:
            raise ProtocolError('failed model load is not retried: ' + self.load_failure)
        check_deadline(self.admission)
        spec = importlib.util.spec_from_file_location('b80_admitted_training_loader', self.loader_path)
        loader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loader)
        tokenizer = loader._load_tokenizer(self.admission['base_model_dir'])
        runtime = loader._runtime_loader(self.root, self.admission, tokenizer)
        torch, model = runtime['torch'], runtime['model']
        named = validate_parameters(model, torch, cast=True)
        verify_loaded_adapter(model, self.original, torch)
        load_file = importlib.import_module('safetensors.torch').load_file
        weights = {arm: load_file(str(path / 'adapter_model.safetensors'), device='cpu')
                   for arm, path in (('A', self.original), ('B', self.child))}
        file_names = {name.replace('.default.weight', '.weight') for name in ORIGINAL_LORA_NAMES}
        for arm in ('A', 'B'):
            require(set(weights[arm]) == file_names and len(weights[arm]) == 80,
                    'both arms must contain exact original 80 tensor names')
            for name, tensor in weights[arm].items():
                require(tensor.dtype == torch.float32 and torch.isfinite(tensor).all().item(),
                        'both adapter files must contain finite FP32 tensors')
                require(tensor.shape == weights['A'][name].shape, 'A/child tensor shape mismatch')
        require(any(not torch.equal(weights['A'][name], weights['B'][name]) for name in file_names),
                'child weights must contain a nonzero actual update')
        for _, parameter in model.named_parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        model.eval()
        model.config.use_cache = True
        self.runtime, self.named, self.weights = runtime, named, weights
        self.modules = {name: module for name, module in model.named_modules() if name in ORIGINAL_LORA_MODULES}
        require(set(self.modules) == ORIGINAL_LORA_MODULES, 'exact 40 original LoRA modules required')
        self.weight_shas = {arm: {name: tensor_sha(weights[arm][name.replace('.default.weight', '.weight')])
                                 for name, _ in named} for arm in ('A', 'B')}
        self.identity['adapter_tensors'] = {arm: {name: dict(shape=list(tensor.shape), dtype=str(tensor.dtype),
                                                          sha256=tensor_sha(tensor))
                                                 for name, tensor in weights[arm].items()} for arm in ('A', 'B')}
        self.current_arm = 'A'

    def _state(self, disabled):
        torch, model = self.runtime['torch'], self.runtime['model']
        route = {}
        for name, module in self.modules.items():
            actual = bool(module.disable_adapters)
            active = list(module.active_adapters)
            require(actual == disabled and active == ['default'], 'actual module adapter route differs')
            route[name] = dict(disabled=actual, active_adapters=active)
        actual_shas = {}
        for name, parameter in model.named_parameters():
            require(not parameter.requires_grad and parameter.grad is None, 'evaluation parameters must all be frozen')
            require(parameter.dtype == (torch.float32 if name in ORIGINAL_LORA_NAMES else torch.bfloat16),
                    'shared base BF16 / LoRA FP32 precision mismatch')
            if name in ORIGINAL_LORA_NAMES:
                actual_shas[name] = tensor_sha(parameter)
        require(actual_shas == self.weight_shas[self.current_arm], 'actual selected adapter values changed')
        require(model.training is False, 'evaluation model must stay in eval mode')
        return dict(arm=self.current_arm, effective_route='base_only' if disabled else self.current_arm,
                    modules=route, tensors_sha256=actual_shas,
                    training=model.training, all_parameters_frozen=True)

    def _select(self, arm):
        require(arm in ('A', 'B'), 'unknown evaluation arm')
        if arm != self.current_arm:
            with self.runtime['torch'].no_grad():
                for name, parameter in self.named:
                    tensor = self.weights[arm][name.replace('.default.weight', '.weight')]
                    parameter.copy_(tensor.to(device=parameter.device, dtype=self.runtime['torch'].float32))
            self.current_arm = arm

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
            require(len(ids) + 512 <= 32768, 'context exceeds 32768 with full 512 generation allowance; no truncation')
            record.update(prompt=prompt, prompt_token_ids=ids, prompt_sha256=sha256_bytes(prompt.encode('utf-8')),
                          prompt_token_ids_sha256=sha256_bytes(raw_bytes(ids)))
            device = model.get_input_embeddings().weight.device
            input_ids = torch.tensor([ids], dtype=torch.long, device=device)
            attention = torch.ones_like(input_ids)
            outer = self._state(False)
            record['route_before'] = outer
            context = model.disable_adapter() if stage != 'Planner' else nullcontext()
            try:
                with context:
                    record['route_enter'] = self._state(stage != 'Planner')
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
                                  route_after_forward=self._state(stage != 'Planner'))
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
