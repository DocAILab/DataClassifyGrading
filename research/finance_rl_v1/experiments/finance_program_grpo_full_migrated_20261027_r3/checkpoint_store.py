"""Exclusive complete optimizer boundaries with exact tensor/RNG recovery."""
import json
import os
from support import put,sha

def save(folder,named,optimizer,*,version,reference_digest):
    import torch
    from pathlib import Path
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    state=dict(weights={n:p.detach().cpu().clone() for n,p in named},optimizer=optimizer.state_dict(),
               policy_version=version,reference_digest=reference_digest,torch_rng=torch.get_rng_state(),
               cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [])
    with (folder/'state.pt').open('xb') as f:torch.save(state,f);f.flush();os.fsync(f.fileno())
    proof=dict(policy_version=version,reference_digest=reference_digest,state_sha256=sha(folder/'state.pt'),tensor_count=len(state['weights']))
    put(folder/'manifest.json',proof);return proof

def restore(folder,named,optimizer,*,reference_digest):
    import torch
    from pathlib import Path
    folder=Path(folder);proof=json.loads((folder/'manifest.json').read_bytes())
    if proof['reference_digest']!=reference_digest or sha(folder/'state.pt')!=proof['state_sha256']:raise ValueError('checkpoint identity differs')
    state=torch.load(folder/'state.pt',map_location='cpu',weights_only=True)
    if state['reference_digest']!=reference_digest or set(state['weights'])!={n for n,_ in named}:raise ValueError('checkpoint tensor/reference identity differs')
    with torch.no_grad():
        for n,p in named:p.copy_(state['weights'][n].to(device=p.device,dtype=p.dtype))
    optimizer.load_state_dict(state['optimizer']);torch.set_rng_state(state['torch_rng'])
    if state['cuda_rng']:torch.cuda.set_rng_state_all(state['cuda_rng'])
    from migration_contract import verify_actual_restore
    actual=verify_actual_restore(named,optimizer,state,torch)
    return dict(proof,actual_restore_validation=actual)

