"""Explicit reentrant checkpoint offload and bounded activation lifecycles.

The context must be entered INSIDE the callable run by checkpoint: a context
around non-reentrant checkpoint cannot own its recomputation saved tensors.
No Torch/HF functions or installed sources are replaced by this module.
"""
from contextlib import contextmanager
import inspect
from pathlib import Path
import re

from io_utils import put


def cpu_offloaded_checkpoint(function, *args, **kwargs):
    """Run the original callable and its backward recompute under CPU hooks.

    Like HF's reentrant checkpoint, differentiable inputs must be positional
    tensors. Non-tensor positional/keyword arguments are retained in the closure.
    The model's frozen embedding output is made differentiable by input_gradients.
    """
    import torch
    from torch.utils.checkpoint import checkpoint

    if kwargs.pop('use_reentrant', True) is not True:
        raise RuntimeError('CPU activation offload requires reentrant checkpoint')
    preserve_rng_state = kwargs.pop('preserve_rng_state', True)
    if torch.is_grad_enabled() and not any(isinstance(x, torch.Tensor) and x.requires_grad for x in args):
        raise RuntimeError('reentrant checkpoint requires a positional input tensor with gradients')

    def run_function(*inputs):
        with torch.autograd.graph.save_on_cpu(pin_memory=True):
            return function(*inputs, **kwargs)

    return checkpoint(run_function, *args, use_reentrant=True, preserve_rng_state=preserve_rng_state)


def configure_gradient_checkpointing(model, expected_decoder_layers=32):
    """Use the modern HF setter's explicit checkpoint-function argument.

    The public enable method hard-codes a partial of Torch checkpoint. The modern
    HF setter is the extension point taking gradient_checkpointing_func; legacy
    setters and incomplete/misrouted application are rejected before any update.
    """
    setter = getattr(model, '_set_gradient_checkpointing', None)
    if not callable(setter) or not {'enable', 'gradient_checkpointing_func'}.issubset(inspect.signature(setter).parameters):
        raise RuntimeError('unsupported HF checkpoint function configuration')
    targets = [(name, module) for name, module in model.named_modules()
               if re.search(r'(?:^|\.)language_model\.layers\.\d+$', name)]
    if len(targets) != expected_decoder_layers or any(not hasattr(module, 'gradient_checkpointing') for _, module in targets):
        raise RuntimeError(f'expected {expected_decoder_layers} language decoder checkpoint targets')
    try:
        setter(enable=True, gradient_checkpointing_func=cpu_offloaded_checkpoint)
        if any(not module.gradient_checkpointing or
               getattr(module, '_gradient_checkpointing_func', None) is not cpu_offloaded_checkpoint for _, module in targets):
            raise RuntimeError('HF checkpoint function was not applied to every decoder target')
    except Exception:
        model.gradient_checkpointing_disable()
        raise
    return {'mechanism': 'reentrant_function_scoped_save_on_cpu', 'use_reentrant': True,
            'decoder_checkpoint_targets': len(targets), 'decoder_target_names': [name for name, _ in targets]}


@contextmanager
def input_gradients(model):
    """Own one removable embedding hook, including on exceptional exit.

    Register directly instead of HF/PEFT's convenience method, which can replace
    its stored handle and orphan earlier hooks on consecutive updates.
    """
    embedding = model.get_input_embeddings()
    def require_grad(module, inputs, output):
        output.requires_grad_(True)
    handle = embedding.register_forward_hook(require_grad)
    try:
        yield
    finally:
        handle.remove()


def memory_snapshot(torch, model, include_summary=False):
    """Read the actual placement and all visible CUDA allocator counters."""
    mapping = getattr(model, 'hf_device_map', None)
    value = {'hf_device_map': {str(k): v if isinstance(v, (str, int)) else str(v)
                               for k, v in mapping.items()} if mapping is not None else None,
             'cuda_available': bool(torch.cuda.is_available()), 'gpus': []}
    if include_summary:
        value['memory_summary'] = {}
    if value['cuda_available']:
        for device in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(device)
            value['gpus'].append({'device': device, 'allocated': torch.cuda.memory_allocated(device),
                                  'reserved': torch.cuda.memory_reserved(device),
                                  'peak_allocated': torch.cuda.max_memory_allocated(device),
                                  'peak_reserved': torch.cuda.max_memory_reserved(device),
                                  'free': free, 'total': total})
            if include_summary:
                value['memory_summary'][str(device)] = torch.cuda.memory_summary(device=device, abbreviated=False)
    return value


def write_memory_event(output, torch, model, event, policy_version, *, record=None,
                       trajectory_index=None, call_index=None, extra=None):
    """Exclusive JSON writes containing only sizes, indices, placement, counters."""
    value = memory_snapshot(torch, model, include_summary=event == 'oom')
    value.update(event=event, policy_version=policy_version, update_index=policy_version + 1,
                 trajectory_index=trajectory_index, call_index=call_index)
    if record is not None:
        value.update(prompt_tokens=len(record['prompt_ids']), output_tokens=len(record['generated_ids']))
    if extra:
        value.update(extra)
    name = ('trajectory-%04d-call-%04d-' % (trajectory_index + 1, call_index + 1)
            if trajectory_index is not None else '') + event + '.json'
    path = Path(output) / 'memory' / name
    put(path, value)
    return path
