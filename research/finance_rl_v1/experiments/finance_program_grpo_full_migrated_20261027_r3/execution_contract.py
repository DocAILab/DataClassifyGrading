"""Immutable admission, sampled-policy settings and exclusive monotone claims."""
import json
import time
from pathlib import Path
from support import put

def options(stage,deadline,*,now=None,stochastic=True):
    remaining=deadline-(time.time() if now is None else now)
    if remaining<=20:raise TimeoutError('absolute RL window exhausted')
    result=dict(do_sample=False,max_new_tokens=512,max_time=min(90.,remaining-10),eos_token_id=[248044,248046],use_cache=True)
    if stage=='Planner' and stochastic:result.update(do_sample=True,temperature=.8,top_p=1.,top_k=0,repetition_penalty=1.)
    return result

class Claims:
    def __init__(self,folder,deadline,limit,carry_in=0):
        if not isinstance(carry_in,int) or not 0<=carry_in<=limit:raise ValueError('valid original consumed budget required')
        self.folder=Path(folder);self.folder.mkdir(exist_ok=True,parents=True);self.deadline=deadline;self.limit=limit;self.carry_in=carry_in
    def claim(self,kind,destination,version,*,now=None):
        now=time.time() if now is None else now
        if now>=self.deadline-20:raise TimeoutError('absolute RL deadline; no reset')
        count=self.carry_in+len(list(self.folder.glob('*.json')))+1
        if count>self.limit:raise RuntimeError('RL total generation/forward call budget exhausted')
        put(self.folder/f'{count:06d}.json',dict(index=count,kind=kind,destination=str(destination),policy_version=version,time_unix=now))
        return count

def validate_group(rows,size=4):
    if len(rows)!=size or len({r['source_id'] for r in rows})!=1 or len({r['policy_version'] for r in rows})!=1 or sorted(r['sample_index'] for r in rows)!=list(range(size)):
        raise ValueError('complete source-isolated fixed-policy group required')
