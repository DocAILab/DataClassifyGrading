"""Verify the accepted actor checkpoint before enabling GRPO-only inference."""
import hashlib,json
from pathlib import Path

def load_checkpoint(folder,expected_sha,reference_digest,torch_module=None):
    if torch_module is None:import torch as torch_module
    folder=Path(folder);manifest=json.loads((folder/'manifest.json').read_bytes());path=folder/'state.pt'
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    if h.hexdigest()!=expected_sha or manifest['state_sha256']!=expected_sha:raise ValueError('accepted checkpoint SHA mismatch')
    if manifest['reference_digest']!=reference_digest or manifest['tensor_count']!=80:raise ValueError('reference or tensor count mismatch')
    state=torch_module.load(path,map_location='cpu',weights_only=True)
    if state['policy_version']!=manifest['policy_version'] or state['reference_digest']!=reference_digest:raise ValueError('checkpoint metadata mismatch')
    if len(state['weights'])!=80:raise ValueError('exactly 80 weights required')
    for v in state['weights'].values():
        if v.is_cuda or v.dtype!=torch_module.float32 or not torch_module.isfinite(v).all().item():raise ValueError('finite CPU FP32 weights required')
    return state

def apply_checkpoint(backend,state,tensor_sha,torch_module,expected_sha):
    weights=state['weights'];named=dict(backend.named)
    if set(named)!=set(weights) or len(named)!=80:raise ValueError('adapter names do not match accepted checkpoint')
    if any(tuple(named[k].shape)!=tuple(v.shape) or named[k].dtype!=v.dtype for k,v in weights.items()):raise ValueError('adapter shape or dtype mismatch')
    with torch_module.no_grad():
        for name,p in named.items():p.copy_(weights[name].to(device=p.device,dtype=p.dtype))
    backend.runtime['model'].eval()
    for _,p in backend.runtime['model'].named_parameters():p.requires_grad_(False);p.grad=None
    backend.weight_shas={name:tensor_sha(p) for name,p in named.items()}
    expected={name:tensor_sha(v) for name,v in weights.items()}
    if backend.weight_shas!=expected:raise ValueError('actual loaded weights differ from accepted checkpoint')
    backend.identity.update(policy_version=state['policy_version'],evaluation_condition='GRPO',
                            active_actor_checkpoint_state_sha256=expected_sha)
    return expected
