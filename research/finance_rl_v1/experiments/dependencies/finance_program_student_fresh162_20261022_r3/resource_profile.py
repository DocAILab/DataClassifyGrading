"""Three-card shared admission with explicit caps and actual 2GiB headroom."""
import itertools
GIB=1024**3
PINNED_UUIDS={0:'GPU-4e00b91b-5123-d78d-35f3-e72212f3f588',2:'GPU-4c58bc24-4fe1-36cb-55ce-e43f346d954c',3:'GPU-fa094e6f-f139-fc8f-dfc4-013092b3a716'}
def candidates(rows):
    eligible=[r for r in rows if r['index'] in PINNED_UUIDS and r['uuid']==PINNED_UUIDS[r['index']] and r['bf16_compatible'] and r['free_bytes']>=30*GIB]
    result=[]
    for combo in itertools.combinations(sorted(eligible,key=lambda r:r['index']),3):
        result.append(dict(gpu_indices=[r['index'] for r in combo],gpu_uuids=[r['uuid'] for r in combo],
            allocator_limit_bytes=[28*GIB]*3,model_max_memory_bytes=[8*GIB]*3,headroom_bytes=2*GIB,
            snapshot=combo,occupied_card_count=sum(bool(r['other_compute_pids']) for r in combo)))
    return sorted(result,key=lambda r:(r['occupied_card_count'],r['gpu_indices']))
def recheck(rows,anchor):
    matches=[c for c in candidates(rows) if c['gpu_indices']==anchor['gpu_indices'] and c['gpu_uuids']==anchor['gpu_uuids']]
    if len(matches)!=1:raise ValueError('locked actual UUID/free memory changed')
    return matches[0]
