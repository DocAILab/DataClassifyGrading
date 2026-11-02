"""Finite inference-only context window; training and model config stay unchanged."""
import json
from pathlib import Path

CONTEXT_TOKENS=49152
GENERATION_TOKENS=512

def validate_context_size(input_tokens,admission):
    if type(input_tokens) is not int or input_tokens<=0:
        raise ValueError('positive integer input token count required')
    if (admission.get('context_window_tokens')!=CONTEXT_TOKENS or
            admission.get('max_sequence_tokens')!=CONTEXT_TOKENS):
        raise ValueError('only frozen49152 inference context is admitted')
    cfg=json.loads((Path(admission['base_model_dir'])/'config.json').read_bytes())
    native=cfg.get('text_config',cfg).get('max_position_embeddings')
    if type(native) is not int or native<CONTEXT_TOKENS:
        raise ValueError('actual native model context does not support49152')
    if input_tokens+GENERATION_TOKENS>CONTEXT_TOKENS:
        raise ValueError('context exceeds49152 including full512 generation allowance; no truncation')
    return CONTEXT_TOKENS
