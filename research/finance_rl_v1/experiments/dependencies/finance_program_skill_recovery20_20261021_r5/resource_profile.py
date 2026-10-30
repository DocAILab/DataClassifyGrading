"""User-selected physical pair only; bounded inference, never training."""
import importlib.util
from pathlib import Path

GIB = 1024 ** 3
UUIDS = ['GPU-4c58bc24-4fe1-36cb-55ce-e43f346d954c',
         'GPU-4e00b91b-5123-d78d-35f3-e72212f3f588']
path = Path(__file__).resolve().parent.parent / 'finance_program_student_b80_eval_20261021_r2/resource_gate.py'
spec = importlib.util.spec_from_file_location('pinned_resource_validation', path)
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)

def candidates(rows, *, max_sequence_tokens):
    if type(max_sequence_tokens) is not int or not 0 < max_sequence_tokens <= 49152:
        raise ValueError('finite untruncated 49152 context required')
    indexed = {r['index']: r for r in native._validated(rows)}
    chosen = [indexed.get(i) for i in (2, 0)]
    if any(r is None or r['uuid'] != u or r.get('bf16_compatible') is not True
           or r['free_bytes'] < 22*GIB for r, u in zip(chosen, UUIDS)):
        return []
    return [dict(gpu_indices=[2, 0], gpu_uuids=UUIDS[:], allocator_limit_bytes=[20*GIB]*2,
                 model_max_memory_bytes=[12*GIB]*2, headroom_bytes=2*GIB,
                 max_sequence_tokens=max_sequence_tokens, admission_kind='user_pair_shared_inference_provisional',
                 occupied_card_count=sum(bool(r['other_compute_pids']) for r in chosen),
                 snapshot=[dict(r) for r in chosen], scientific_updates_allowed=0)]

def recheck(rows, anchor):
    selected = candidates(rows, max_sequence_tokens=anchor['max_sequence_tokens'])
    keys = ('gpu_indices', 'gpu_uuids', 'allocator_limit_bytes', 'model_max_memory_bytes', 'headroom_bytes')
    if len(selected) != 1 or any(selected[0][k] != anchor[k] for k in keys):
        raise ValueError('locked selected pair identity/headroom changed')
    return selected[0]
