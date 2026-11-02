"""Identity and whole-optimizer boundaries for the single step08 continuation."""
import math
import time

def validate_resume(meta,rows,parent_freeze_sha,train_sha,admission,now=None):
    identity=meta['identity'];origin=identity['admission']
    if identity['freeze_sha256']!=parent_freeze_sha or identity['train_sha256']!=train_sha:
        raise ValueError('parent checkpoint identity mismatch')
    if meta['schema']!=1 or meta['step']!=8 or len(rows)!=162:
        raise ValueError('only saved step8 boundary may resume')
    if meta['consumed']!=[r['id'] for r in rows[:128]] or len(set(meta['consumed']))!=128:
        raise ValueError('checkpoint consumed prefix differs')
    if any(admission[k]!=origin[k] for k in ('gpu_started_unix','gpu_deadline_unix')):
        raise ValueError('original absolute clock must not reset')
    if (time.time() if now is None else now)>=origin['gpu_deadline_unix']:
        raise ValueError('original absolute clock expired')
    return 128

def remaining_batches(rows,step):
    if len(rows)!=162 or step!=8:raise ValueError('only original162 step8 may resume')
    return [(9,list(rows[128:144])),(10,list(rows[144:160])),(11,list(rows[160:162]))]

def verify_restored(named,optimizer,manifest,torch):
    if [n for n,_ in named]!=manifest['parameter_names'] or len(named)!=80:
        raise ValueError('restored80 parameter names differ')
    if len(optimizer.state)!=80:raise ValueError('restored80 AdamW states required')
    for n,p in named:
        if p.dtype!=torch.float32 or not bool(torch.isfinite(p).all()):raise ValueError('restored finiteFP32 parameter required')
        st=optimizer.state[p]
        if float(st['step'])!=8:raise ValueError('restored AdamW step differs')
        for key in ('exp_avg','exp_avg_sq'):
            v=st[key]
            if v.shape!=p.shape or v.dtype!=torch.float32 or not bool(torch.isfinite(v).all()):
                raise ValueError('restored AdamW moment differs')
    return dict(parameters=80,optimizer_states=80,optimizer_step=8,finite_fp32=True)
