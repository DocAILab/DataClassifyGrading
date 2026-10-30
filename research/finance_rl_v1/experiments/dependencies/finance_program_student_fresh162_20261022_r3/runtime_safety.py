"""Fail closed on shared-resource pressure; persist complete optimizer boundaries."""
from pathlib import Path
import shutil
from recovery_safety import require_host_headroom,save_checkpoint
GIB=1024**3

def validate_resources(host_available,gpu_free,disk_free,*,preload=False):
    require_host_headroom(available_bytes=host_available,required_bytes=(96 if preload else 64)*GIB)
    if len(gpu_free)!=3 or min(gpu_free)<2*GIB:raise RuntimeError('GPU actual free below 2GiB; safe stop, no truncation')
    if disk_free<20*GIB:raise RuntimeError('checkpoint disk reserve below 20GiB')

def guard_resources(root,torch=None,*,preload=False):
    info={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.split(':')[0] in ('MemAvailable',)}
    free=[torch.cuda.mem_get_info(i)[0] for i in range(3)] if torch is not None else [2*GIB]*3
    validate_resources(info['MemAvailable'],free,shutil.disk_usage(root).free,preload=preload)
    return dict(host_available_bytes=info['MemAvailable'],gpu_free_bytes=free)

def persist_boundary(root,named,optimizer,*,step,consumed,identity):
    optimizer.zero_grad(set_to_none=True)
    return save_checkpoint(root,named,optimizer,step=step,consumed=consumed,identity=identity)
