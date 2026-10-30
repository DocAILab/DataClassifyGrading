"""Release inactive pinned-host allocations after a completed backward pass."""
import gc

def trim_host_cache(torch):
    empty = getattr(torch._C, '_host_emptyCache', None)
    if not callable(empty):
        raise RuntimeError('installed PyTorch lacks validated host-cache release API')
    gc.collect()
    for device in range(torch.cuda.device_count()):
        torch.cuda.synchronize(device)
    before = torch.cuda.memory.host_memory_stats()
    empty()
    after = torch.cuda.memory.host_memory_stats()
    return dict(before=before, after=after)
