"""Dynamic UUID admission; release only unused CUDA cache, preserve 2GiB guard."""
import json,time,itertools,gc,os
from pathlib import Path
from support import ROOT,TRAIN,put,load
GIB=1024**3

def candidates(rows,p):
 cap=int(p['gpu_cap_GiB']*GIB);head=int(p['gpu_reserve_GiB']*GIB);threshold=int(p['gpu_admission_free_GiB']*GIB)
 if cap>28*GIB or head<2*GIB or threshold<cap+head:raise ValueError('bounded cap and original reserve required')
 eligible=[v for v in rows if str(v['index']) in p['allowed_gpu_uuids'] and p['allowed_gpu_uuids'][str(v['index'])]==v['uuid'] and v['bf16_compatible'] and v['free_bytes']>=threshold]
 result=[]
 for cc in itertools.combinations(sorted(eligible,key=lambda r:r['index']),3):
  result.append(dict(gpu_indices=[v['index'] for v in cc],gpu_uuids=[v['uuid'] for v in cc],allocator_limit_bytes=[cap]*3,model_max_memory_bytes=[8*GIB]*3,headroom_bytes=head,snapshot=cc))
 return sorted(result,key=lambda c:(-min(v['free_bytes'] for v in c['snapshot']),sum(bool(v['other_compute_pids']) for v in c['snapshot']),c['gpu_indices']))

def recheck(rows,anchor,p):
 found=[v for v in candidates(rows,p) if v['gpu_indices']==anchor['gpu_indices'] and v['gpu_uuids']==anchor['gpu_uuids']]
 if len(found)!=1:raise ValueError('locked UUID/free changed')
 return found[0]

def cuda_snapshot(torch):
 result=[]
 for i in range(3):
  free,total=torch.cuda.mem_get_info(i)
  result.append(dict(logical_device=i,free_bytes=free,total_bytes=total,allocated_bytes=torch.cuda.memory_allocated(i),reserved_bytes=torch.cuda.memory_reserved(i),peak_allocated_bytes=torch.cuda.max_memory_allocated(i),peak_reserved_bytes=torch.cuda.max_memory_reserved(i)))
 return result

def release_cache(torch,before):
 if not any(v['reserved_bytes']-v['allocated_bytes']>GIB//4 or v['free_bytes']<4*GIB for v in before):return False
 gc.collect()
 for i in range(3):
  with torch.cuda.device(i):torch.cuda.empty_cache()
 return True

def resources(backend):
 import sys
 if str(TRAIN) not in sys.path:sys.path.append(str(TRAIN))
 from runtime_safety import guard_resources
 torch=backend.runtime['torch'] if backend.runtime else None
 if torch is None:return guard_resources(ROOT)
 before=cuda_snapshot(torch);reclaimed=release_cache(torch,before)
 after=cuda_snapshot(torch);v=dict(time_unix=time.time(),attempt_id=__import__('os').environ.get('FINANCE_ATTEMPT_ID'),policy_version=backend.version,before=before,after=after,reclaimed_unused_cache=reclaimed)
 try:value=guard_resources(ROOT,torch)
 except BaseException as e:
  put(ROOT/f'runs/resource-checks/{time.time_ns()}.json',dict(v,error=str(e)));raise
 put(ROOT/f'runs/resource-checks/{time.time_ns()}.json',dict(v,passed=True))
 return value

def publish_reference(path,value):
 p=Path(path)
 if p.exists():
  old=json.loads(p.read_bytes())
  if old['tensors']!=value['tensors'] or old['reference_digest']!=value['reference_digest'] or old['initial_adapter_sha256']!=value['initial_adapter_sha256']:raise ValueError('frozen original reference changed across attempts')
 else:put(p,value)
 attempt=int(os.environ.get('FINANCE_ATTEMPT_ID','0'))
 put(ROOT/f'runs/references/attempt-{attempt:03d}.json',value)
