"""Shared sum-then-normalize advantages and reference scalar objective."""
import math
import statistics


def advantages(groups, outcomes, epsilon=1e-8):
    if len(groups)!=len(outcomes) or not groups: raise ValueError('aligned nonempty groups required')
    if any(len(r)!=len(o) or len(r)<2 for r,o in zip(groups,outcomes)):
        raise ValueError('at least two episodes per complete group required')
    if any(not math.isfinite(x) for g in groups for x in g):raise ValueError('nonfinite reward')
    active=[[any(o) for _ in r] for r,o in zip(groups,outcomes)]
    retained=[x for r,a in zip(groups,active) for x,keep in zip(r,a) if keep]
    scale=statistics.pstdev(retained) if retained else 0.
    aa=[]
    for r,a in zip(groups,active):
        mean=statistics.mean(r)
        aa.append([(x-mean)/(scale+epsilon) if keep and scale>epsilon else 0. for x,keep in zip(r,a)])
    return dict(advantages=aa,active=active,batch_std=scale,
                all_wrong_groups=sum(not any(o) for o in outcomes),
                zero_advantage_groups=sum(all(x==0 for x in r) for r in aa))


def scalar_loss(current, old, reference, advantage, mask, *, clip=.2,beta=.02):
    if len({len(x) for x in (current,old,reference,advantage,mask)})!=1:raise ValueError('aligned token arrays')
    terms=[]
    for p,q,r,a,m in zip(current,old,reference,advantage,mask):
        if not m:continue
        ratio=math.exp(p-q);clipped=min(max(ratio,1-clip),1+clip)
        delta=r-p;kl=math.exp(delta)-delta-1
        terms.append(-min(ratio*a,clipped*a)+beta*kl)
    return sum(terms)/len(terms) if terms else 0.

