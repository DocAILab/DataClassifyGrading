"""GPU0-only inference envelope, no training permission or clock reset."""
import importlib.util
import time
from pathlib import Path
GIB=1024**3
GPU0='GPU-4e00b91b-5123-d78d-35f3-e72212f3f588'
DEADLINE=1792587659.6128213
path=Path(__file__).resolve().parent.parent/'finance_program_student_b80_eval_20261021_r2/resource_gate.py'
s=importlib.util.spec_from_file_location('original_snapshot_validation',path)
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def candidates(rows,*,max_sequence_tokens):
 if type(max_sequence_tokens) is not int or not 0<max_sequence_tokens<=49152:raise ValueError('finite48K actual input proof required')
 rows=m._validated(rows)
 eligible=[r for r in rows if r['index']==0 and r['uuid']==GPU0 and r.get('bf16_compatible') is True and r['free_bytes']>=22*GIB]
 return [dict(gpu_indices=[0],gpu_uuids=[GPU0],allocator_limit_bytes=[20*GIB],model_max_memory_bytes=[18*GIB],headroom_bytes=2*GIB,max_sequence_tokens=max_sequence_tokens,admission_kind='gpu0_text_only_visual_cpu_provisional',snapshot=[dict(r)],occupied_card_count=int(bool(r['other_compute_pids'])),scientific_updates_allowed=0) for r in eligible]
def recheck(rows,anchor):
 match=candidates(rows,max_sequence_tokens=anchor['max_sequence_tokens'])
 if len(match)!=1 or any(match[0][k]!=anchor[k] for k in ['gpu_indices','gpu_uuids','allocator_limit_bytes','model_max_memory_bytes','headroom_bytes']):raise ValueError('locked GPU0 identity/memory changed')
 return match[0]
def validate_admission(a,now=None):
 if (a.get('gpu_indices')!=[0] or a.get('gpu_uuids')!=[GPU0] or a.get('allocator_limit_bytes')!=[20*GIB] or a.get('model_max_memory_bytes')!=[18*GIB] or a.get('gpu_deadline_unix')!=DEADLINE or a.get('max_sequence_tokens')!=49152 or a.get('context_window_tokens')!=49152):raise ValueError('only locked GPU0 finite48K original deadline admission permitted')
 if (time.time() if now is None else now)>=DEADLINE:raise ValueError('original absolute stage deadline expired')
