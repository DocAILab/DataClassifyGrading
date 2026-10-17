"""Three unlabelled bounded engineering smokes through the frozen Program core."""
import argparse
from copy import deepcopy
from importlib.metadata import PackageNotFoundError, distribution
import json
import os
from pathlib import Path
import platform
import sys
import time

from download_model import sha256, verify_model, write_json

OFFLINE = {'local_files_only': True, 'trust_remote_code': False}
CONTEXT_CAP = 32768
OUTPUT_CAP = 512
SEED = 20260928


def generation_metadata(result):
    """Bounded, observed per-call fields shared by execution and CPU replay."""
    return {key: deepcopy(result[key]) for key in
            ('input_tokens', 'output_tokens', 'truncation_at_budget', 'error') if key in result}


class ContextBudgetExceeded(ValueError):
    def __init__(self, tokens):
        self.input_tokens = tokens
        super().__init__(f'context budget exceeded: {tokens}+{OUTPUT_CAP}>{CONTEXT_CAP}; no truncation')


def reserve_run(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir()  # Atomic directory creation; must precede any CUDA allocation.
    return path


def smoke_inputs():
    definitions = [('account_balance', '账户余额'), ('contact_phone', '客户联系手机号'),
                   ('invoice_amount', '发票金额'), ('customer_name', '客户姓名'),
                   ('order_id', '订单标识'), ('transaction_date', '交易日期')]
    registry = {name: {'label': desc, 'description': desc, 'grading_level': 'L2'} for name, desc in definitions}
    rows = [{'id': ident, 'domain': '金融', 'metadata': {'field_name': name, 'field_description': desc}}
            for ident, name, desc in [('local-balance', 'account_balance', '账户当前余额'),
                                     ('local-phone', 'customer_phone', '客户联系手机号'),
                                     ('local-invoice', 'invoice_amount', '发票金额')]]
    return registry, rows


def generation_options(config, tokenizer):
    eos = getattr(config, 'eos_token_id', None)
    if eos is None:
        eos = getattr(tokenizer, 'eos_token_id', None)
    if eos is None:
        raise ValueError('actual model/tokenizer EOS is required')
    pad = getattr(tokenizer, 'pad_token_id', None)
    if pad is None:
        pad = getattr(config, 'pad_token_id', None)
    if pad is None:
        pad = eos[0] if isinstance(eos, list) else eos
    return {'do_sample': False, 'max_new_tokens': OUTPUT_CAP, 'max_time': 90.0,
            'eos_token_id': eos, 'pad_token_id': pad}


def encode_messages(tokenizer, messages):
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    encoded = tokenizer(prompt, return_tensors='pt', truncation=False)
    tokens = encoded['input_ids'].shape[1]
    if encoded['input_ids'].shape[0] != 1:
        raise ValueError('batch size must be one')
    if tokens + OUTPUT_CAP > CONTEXT_CAP:
        raise ContextBudgetExceeded(tokens)
    return encoded


def environment(model_path, *, include_comparison_explanation=True):
    packages = {}
    for package in ('torch', 'transformers', 'accelerate', 'huggingface_hub', 'safetensors'):
        try:
            packages[package] = distribution(package).metadata.get('Version')
        except PackageNotFoundError:
            packages[package] = None
    root = Path(__file__).resolve().parent
    from tools.freeze_revision import environment_identity
    identity = environment_identity(root)
    return {'include_comparison_explanation': include_comparison_explanation,
            'pid': os.getpid(), 'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
            'packages': packages, 'model_path': str(Path(model_path).resolve()), **identity,
            'seed': SEED, 'context_cap': CONTEXT_CAP, 'max_new_tokens': OUTPUT_CAP, 'per_generate_max_time': 90.0,
            'whole_process_timeout_seconds': 900, 'whole_process_timeout_enforced_by': 'external coordinator',
            'inference': 'FP16 cuda:0 batch1 greedy enable_thinking=False'}


class Backend:
    def __init__(self, model_path, call_dir):
        self.call_dir = Path(call_dir)
        self.call_dir.mkdir(parents=True, exist_ok=True)
        self.calls = []
        self.failures = []
        self.identity = verify_model(model_path)
        import torch
        from transformers import AutoConfig, AutoModelForImageTextToText, AutoTokenizer
        self.torch = torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is required; no CPU/offload fallback')
        torch.cuda.set_device(0)
        free, total = torch.cuda.mem_get_info(0)
        if free <= 0:
            raise RuntimeError('cuda:0 has no free device memory')
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        config = AutoConfig.from_pretrained(str(model_path), **OFFLINE)
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), **OFFLINE)
        if config.model_type != 'qwen3_5':
            raise ValueError(f'unsupported official checkpoint config: {config.model_type}')
        self.attention = 'sdpa' if hasattr(torch.nn.functional, 'scaled_dot_product_attention') else 'eager'
        self.model = AutoModelForImageTextToText.from_pretrained(str(model_path), **OFFLINE, dtype=torch.float16,
                     device_map={'': 'cuda:0'}, attn_implementation=self.attention)
        self.model.eval()
        self.parameter_devices = sorted({str(p.device) for p in self.model.parameters()})
        self.parameter_dtypes = sorted({str(p.dtype) for p in self.model.parameters() if p.is_floating_point()})
        if self.parameter_devices != ['cuda:0'] or self.parameter_dtypes != ['torch.float16']:
            raise RuntimeError(f'checkpoint must be entirely FP16 cuda:0: {self.parameter_devices}, {self.parameter_dtypes}')
        self.options = generation_options(self.model.generation_config, self.tokenizer)
        write_json(self.call_dir / 'backend.json', {'repo_id': self.identity.get('repo_id'), 'revision': self.identity['revision'],
                   'model_class': type(self.model).__name__, 'config_type': config.model_type,
                   'attention_implementation': self.attention, 'parameter_devices': self.parameter_devices,
                   'parameter_dtypes': self.parameter_dtypes, 'cuda_device': torch.cuda.get_device_name(0),
                   'torch_cuda_version': torch.version.cuda, 'free_bytes_before_load': free, 'total_device_bytes': total,
                   'generation': self.options, 'offline_options': OFFLINE})

    def __call__(self, messages):
        return self.generate(messages)

    def generate(self, messages):
        self.last_generation_metadata = {}
        call = self.call_dir / f'call-{len(self.calls) + 1:04d}'
        call.mkdir()
        self.calls.append(str(call))
        started = time.perf_counter()
        result = {'messages': deepcopy(messages), 'raw_text': None, 'input_tokens': None, 'output_tokens': None,
                  'input_devices': [], 'parameter_devices': self.parameter_devices, 'parameter_dtypes': self.parameter_dtypes,
                  'attention_implementation': self.attention, 'generation': self.options,
                  'truncation_at_budget': False, 'error': None}
        write_json(call / 'request.json', {'messages': messages, 'generation': self.options})
        try:
            encoded = encode_messages(self.tokenizer, messages)
            result['input_tokens'] = encoded['input_ids'].shape[1]
            encoded = {key: value.to('cuda:0') for key, value in encoded.items()}
            result['input_devices'] = sorted({str(value.device) for value in encoded.values()})
            if result['input_devices'] != ['cuda:0']:
                raise RuntimeError(f'input tensors must be on cuda:0: {result["input_devices"]}')
            self.torch.cuda.reset_peak_memory_stats(0)
            self.torch.cuda.synchronize(0)
            with self.torch.inference_mode():
                output = self.model.generate(**encoded, **self.options)
            self.torch.cuda.synchronize(0)
            generated = output[0, result['input_tokens']:]
            result['output_tokens'] = len(generated)
            result['truncation_at_budget'] = len(generated) >= OUTPUT_CAP
            result['raw_text'] = self.tokenizer.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            return result['raw_text']
        except Exception as exc:
            if isinstance(exc, ContextBudgetExceeded):
                result['input_tokens'] = exc.input_tokens
            result['error'] = f'{type(exc).__name__}: {exc}'
            self.failures.append({'call': str(call), 'error': result['error']})
            raise
        finally:
            result['seconds'] = time.perf_counter() - started
            result['memory'] = {}
            result['telemetry_errors'] = {}
            for name in ('memory_allocated', 'memory_reserved', 'max_memory_allocated'):
                try:
                    result['memory'][name] = getattr(self.torch.cuda, name)(0)
                except Exception as exc:
                    result['memory'][name] = None
                    result['telemetry_errors'][name] = f'{type(exc).__name__}: {exc}'
            self.last_generation_metadata = generation_metadata(result)
            write_json(call / 'result.json', result)


def main(argv=None, *, backend_factory=Backend):
    parser = argparse.ArgumentParser(description='Three unlabelled local GPU engineering smokes; no accuracy metric')
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--include-comparison-explanation', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    try:
        output = reserve_run(args.output)
    except Exception as exc:
        print(f'{type(exc).__name__}: {exc}', file=sys.stderr)
        return 2
    started = time.perf_counter()
    summary = {'status': 'initialization_failure', 'terminated_count': 0, 'accepted_plans': 0,
               'planner_attempts': 0, 'skill_steps': 0, 'backend_errors': [], 'samples': []}
    try:
        write_json(output / 'environment.json', environment(args.model, include_comparison_explanation=args.include_comparison_explanation))
        (output / 'samples').mkdir()
        backend = backend_factory(args.model, output / 'calls')
        from program_loop import ProgramConfig, build_program_controller
        config = ProgramConfig(max_programs=3, max_program_steps=10, max_skill_steps=12,
                               max_planner_attempts=4, max_identical_calls=1)
        registry, rows = smoke_inputs()
        write_json(output / 'inputs.json', {'registry': registry, 'rows': rows, 'budget': vars(config),
                                           'include_comparison_explanation': args.include_comparison_explanation})
        summary['status'] = 'completed'
        def generate(messages):
            generate.last_generation_metadata = {}
            try:
                return backend(messages)
            except Exception as exc:
                summary['backend_errors'].append({'error': f'{type(exc).__name__}: {exc}'})
                raise
            finally:
                generate.last_generation_metadata = generation_metadata(getattr(backend, 'last_generation_metadata', {}))
        for row in rows:
            controller = build_program_controller(generate, registry, rows, config,
                                                   include_comparison_explanation=args.include_comparison_explanation)
            trace = controller.run({'source_id': row['id'], 'domain': row['domain'], 'field_name': row['metadata']['field_name']})
            trace.execution_mode = 'caller supplied backend I/O + real frozen executor; backend evidence in calls/'
            trace.write_json(output / 'samples' / (row['id'] + '.json'))
            summary['terminated_count'] += int(trace.terminated)
            summary['accepted_plans'] += trace.budget['programs']
            summary['planner_attempts'] += trace.budget['planner_attempts']
            summary['skill_steps'] += trace.budget['skill_steps']
            summary['samples'].append({'id': row['id'], 'terminated': trace.terminated, 'stop_reason': trace.stop_reason,
                     'budget': trace.budget, 'planner_errors': sum('error' in a for a in trace.planner_attempts),
                     'execution_errors': sum(s['status'] != 'success' for p in trace.programs for s in p['steps'])})
        if summary['backend_errors']:
            summary['status'] = 'backend_failure'
        records = [json.loads(p.read_text(encoding='utf-8')) for p in (output / 'calls').glob('call-*/result.json')]
        summary['model_calls'] = len(records)
        summary['input_tokens'] = sum(r.get('input_tokens') or 0 for r in records)
        summary['output_tokens'] = sum(r.get('output_tokens') or 0 for r in records)
        summary['generation_seconds'] = sum(r.get('seconds') or 0 for r in records)
    except Exception as exc:
        if summary['status'] != 'initialization_failure':
            summary['status'] = 'runner_failure'
        summary['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        summary['seconds'] = time.perf_counter() - started
        write_json(output / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=True))
    return 0 if summary['status'] == 'completed' else 2


if __name__ == '__main__':
    raise SystemExit(main())
