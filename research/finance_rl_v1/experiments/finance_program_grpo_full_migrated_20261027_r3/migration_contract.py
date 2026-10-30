"""Explicit user-pause lineage and idempotent runtime configuration recovery."""
import json,os
from pathlib import Path

def publish_configuration(path,value):
    path=Path(path)
    if path.exists():
        if json.loads(path.read_bytes())!=value:raise ValueError('restored runtime configuration differs')
        return False
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f:
        f.write(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()+b'\n')
        f.flush();os.fsync(f.fileno())
    return True

def require_origin_pause(v):
    try:
        okay=(v['paused_by_user'] is True and v['completed'] is False and v['all_training_processes_stopped'] is True
              and v['episodes']==96 and v['actual_complete_updates']==2 and v['third_optimizer_started'] is False
              and v['original_worker_exit']['actual_returncode']==-2 and v['original_launch_exit']['actual_returncode']==1
              and v['original_failure']['type']=='KeyboardInterrupt')
    except (KeyError,TypeError):okay=False
    if not okay:raise ValueError('explicit original user-pause proof required')

def logical_destination(destination,relative,roots):
    return any(destination==r.rstrip('/')+'/'+relative for r in roots)

def policy_for_attempt(attempt,origin,current):
    if int(attempt)<1:raise ValueError('positive immutable attempt id required')
    return origin if int(attempt)==1 else current

def verify_actual_restore(named,optimizer,state,torch):
    if any(not torch.equal(p.detach().cpu(),state['weights'][n]) for n,p in named):raise ValueError('actual GPU weight restore differs')
    actual=optimizer.state_dict();expected=state['optimizer']
    if actual['param_groups']!=expected['param_groups'] or set(actual['state'])!=set(expected['state']):raise ValueError('actual optimizer groups differ')
    for k,entries in actual['state'].items():
        if set(entries)!=set(expected['state'][k]):raise ValueError('actual optimizer state keys differ')
        for name,value in entries.items():
            other=expected['state'][k][name]
            if torch.is_tensor(value):
                if not torch.equal(value.detach().cpu(),other.detach().cpu()):raise ValueError('actual optimizer tensor restore differs')
            elif value!=other:raise ValueError('actual optimizer scalar restore differs')
    if not torch.equal(torch.get_rng_state(),state['torch_rng']):raise ValueError('actual CPU RNG restore differs')
    cuda=torch.cuda.get_rng_state_all() if state['cuda_rng'] else []
    if len(cuda)!=len(state['cuda_rng']) or any(not torch.equal(a.cpu(),b.cpu()) for a,b in zip(cuda,state['cuda_rng'])):raise ValueError('actual logical CUDA RNG restore differs')
    return dict(actual_weights_exact=True,actual_optimizer_exact=True,actual_CPU_RNG_exact=True,actual_logical_CUDA_RNG_exact=bool(cuda),CUDA_devices_verified=len(cuda),tensors_verified=len(named))
