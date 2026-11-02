"""Atomic optimizer-boundary checkpoints. No model loading or automatic restart."""
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile

class HostMemoryWait(RuntimeError):pass

def require_host_headroom(*,available_bytes,required_bytes):
    if available_bytes < 0 or required_bytes <= 0:raise ValueError('invalid memory observation or reserve')
    if available_bytes < required_bytes:
        raise HostMemoryWait(f'host available {available_bytes} < required reserve {required_bytes}; defer, do not truncate')

def _cpu(value,torch):
    if torch.is_tensor(value):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:_cpu(v,torch) for k,v in value.items()}
    if isinstance(value,list):return [_cpu(v,torch) for v in value]
    if isinstance(value,tuple):return tuple(_cpu(v,torch) for v in value)
    return value

def save_checkpoint(root,named,optimizer,*,step,consumed,identity):
    import numpy as np
    import torch
    if step < 0 or len(consumed)!=len(set(consumed)):raise ValueError('invalid checkpoint progress')
    if any(p.grad is not None for _,p in named):raise ValueError('checkpoint only at zeroed optimizer boundary')
    root=Path(root);root.mkdir(parents=True,exist_ok=True);final=root/f'checkpoint-{step:02d}'
    if final.exists():raise FileExistsError(final)
    temporary=Path(tempfile.mkdtemp(prefix=f'.checkpoint-{step:02d}-',dir=root))
    numpy_state=np.random.get_state()
    state=dict(parameters={n:_cpu(p,torch) for n,p in named},optimizer=_cpu(optimizer.state_dict(),torch),
        torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
        python_rng=random.getstate(),numpy_rng=[numpy_state[0],numpy_state[1].tolist(),*numpy_state[2:]])
    with (temporary/'state.pt').open('xb') as file:
        torch.save(state,file);file.flush();os.fsync(file.fileno())
    payload=(temporary/'state.pt').read_bytes()
    manifest=dict(schema=1,step=step,consumed=consumed,identity=identity,
        state_sha256=hashlib.sha256(payload).hexdigest(),state_bytes=len(payload),parameter_names=[n for n,_ in named])
    with (temporary/'manifest.json').open('x',encoding='utf-8') as file:
        json.dump(manifest,file,sort_keys=True);file.flush();os.fsync(file.fileno())
    # Partial writes remain under a hidden temporary name and cannot be admitted.
    os.rename(temporary,final)
    return final

def load_checkpoint(folder,named,optimizer,*,identity):
    import numpy as np
    import torch
    folder=Path(folder)
    if folder.name.startswith('.'):raise ValueError('partial checkpoint')
    manifest=json.loads((folder/'manifest.json').read_bytes())
    if manifest['schema']!=1 or manifest['identity']!=identity:raise ValueError('checkpoint identity differs')
    payload=(folder/'state.pt').read_bytes()
    if len(payload)!=manifest['state_bytes'] or hashlib.sha256(payload).hexdigest()!=manifest['state_sha256']:
        raise ValueError('checkpoint SHA/size differs')
    state=torch.load(folder/'state.pt',map_location='cpu',weights_only=True)
    if manifest['parameter_names']!=[n for n,_ in named] or set(state['parameters'])!={n for n,_ in named}:
        raise ValueError('checkpoint parameter names differ')
    for n,p in named:
        value=state['parameters'][n]
        if value.shape!=p.shape or value.dtype!=p.dtype or not torch.isfinite(value).all().item():
            raise ValueError('checkpoint parameter shape/dtype/finite mismatch')
    if state['cuda_rng'] and (not torch.cuda.is_available() or len(state['cuda_rng'])!=torch.cuda.device_count()):
        raise ValueError('checkpoint CUDA RNG device count differs')
    with torch.no_grad():
        for n,p in named:p.copy_(state['parameters'][n].to(p.device))
    optimizer.load_state_dict(state['optimizer']);optimizer.zero_grad(set_to_none=True)
    random.setstate(state['python_rng']);s=state['numpy_rng']
    np.random.set_state((s[0],np.asarray(s[1],dtype=np.uint32),s[2],s[3],s[4]))
    torch.set_rng_state(state['torch_rng'])
    if state['cuda_rng']:torch.cuda.set_rng_state_all(state['cuda_rng'])
    return manifest
