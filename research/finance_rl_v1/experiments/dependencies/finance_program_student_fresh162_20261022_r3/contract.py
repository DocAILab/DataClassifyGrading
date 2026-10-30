"""New162 identities, full-epoch batching and source/group separation."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
SEED=20261022
MAX_LENGTH=49152
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def raw(v):return json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
def verify_bytes(payload,expected):
    if hashlib.sha256(payload).hexdigest()!=expected:raise ValueError('byte identity mismatch')
    return payload
def put(p,value):
    p=Path(p);p.parent.mkdir(exist_ok=True,parents=True)
    with p.open('xb') as f:
        f.write(raw(value)+b'\n');f.flush();__import__('os').fsync(f.fileno())
def batches(rows):
    if len(rows)!=162:raise ValueError('exact full162 epoch required')
    return [list(rows[i:i+16]) for i in range(0,len(rows),16)]
def divisor(batch):
    if not batch:raise ValueError('empty micro accumulation')
    return len(batch)
def require_isolated(train,evaluation):
    sources={r['source_id'] for r in train};groups={r['group'] for r in train}
    if any(r['source_id'] in sources or r['group'] in groups for r in evaluation):raise ValueError('train/eval source or group overlap')
def fresh_lora_settings(targets):
    if len(targets)!=40:raise ValueError('40 language module targets required')
    return dict(r=8,lora_alpha=16,lora_dropout=.05,bias='none',task_type='CAUSAL_LM',
        target_modules=targets,init_lora_weights=True)
def disable_for(arm,stage):
    if arm not in ('F','N'):raise ValueError('unknown evaluation arm')
    return arm=='F' or stage!='Planner'
def validate_training_counts(d):
    import math
    if d['rows']!=162 or len(d['rows_consumed'])!=162 or len(set(d['rows_consumed']))!=162:raise ValueError('unique actual162 required')
    if d['optimizer_updates']!=11 or d['old_adapters_loaded'] is not False:raise ValueError('new initialization/11 updates required')
    steps=d['optimizer_steps']
    if len(steps)!=11 or [s['step'] for s in steps]!=list(range(1,12)) or [s['normalization_divisor'] for s in steps]!=[16]*10+[2]:raise ValueError('whole epoch step/tail accounting mismatch')
    if any(not s['completed'] or not math.isfinite(s['update_norm']) or s['update_norm']<=0 for s in steps):raise ValueError('finite actual updates required')
def load_train():
    lock=json.loads((ROOT/'data/data-lock.json').read_bytes())
    data=verify_bytes((ROOT/'data/train162.jsonl').read_bytes(),lock['train_sha256'])
    rows=[json.loads(line) for line in data.splitlines()]
    if len(rows)!=162 or len({r['id'] for r in rows})!=162:raise ValueError('162 unique examples required')
    if any(set(r)!={'id','messages','completion'} for r in rows):raise ValueError('training input contains unexpected metadata/reference')
    return rows
