"""Limit temporary GPU memory for long, unpadded Qwen3.5 teacher demonstrations."""
from functools import wraps


def bounded_rule(rule, size=1024):
    from torch.utils.checkpoint import checkpoint
    import torch

    @wraps(rule)
    def apply(query, key, value, g, beta, initial_state=None, output_final_state=False,
              use_qk_l2norm_in_kernel=False, **kwargs):
        if kwargs.get("cu_seqlens") is not None or size % 64:
            raise ValueError("Expected one unpadded sequence and aligned chunk boundaries")

        def block(q, k, v, decay, weight, state):
            return rule(q, k, v, g=decay, beta=weight, initial_state=state, output_final_state=True,
                        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel, **kwargs)

        state, outputs = initial_state, []
        for start in range(0, query.shape[1], size):
            chunk = slice(start, start + size)
            args = [x[:, chunk] for x in (query, key, value, g, beta)]
            # Keep the recurrence differentiable across blocks. Every token contributes
            # to the same objective and each block reuses the native 64-token kernel.
            if torch.is_grad_enabled():
                out, state = checkpoint(block, *args, state, use_reentrant=False)
            else:
                out, state = block(*args, state)
            outputs.append(out)
        return torch.cat(outputs, dim=1), state if output_final_state else None
    return apply


def enable_bounded_delta(model):
    from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule
    count = 0
    for module in model.modules():
        if getattr(module, "chunk_gated_delta_rule", None) is torch_chunk_gated_delta_rule:
            module.chunk_gated_delta_rule = bounded_rule(torch_chunk_gated_delta_rule)
            count += 1
    if not count:
        raise RuntimeError("Expected the native Qwen torch delta-rule backend")
