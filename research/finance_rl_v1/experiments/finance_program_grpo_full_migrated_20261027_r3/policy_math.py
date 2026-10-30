"""Differentiable action-only GRPO math, same completion token boundaries."""
from contextlib import nullcontext

def action_log_probs(model,prompt,response,temperature=1.):
    import torch
    if not prompt or not response or len(prompt)+len(response)>49152 or temperature<=0:
        raise ValueError('exact nonempty prompt/action within context required')
    device=model.get_input_embeddings().weight.device
    ids=torch.tensor([prompt+response],dtype=torch.long,device=device)
    context=torch.autocast(device_type='cuda',dtype=torch.bfloat16) if device.type=='cuda' else nullcontext()
    with context:
        output=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,logits_to_keep=len(response)+1)
    logits=output.logits[:,:-1].float()/temperature
    if logits.shape[1]!=len(response):raise ValueError('causal suffix logits mismatch')
    targets=torch.tensor(response,dtype=torch.long,device=logits.device)
    return -torch.nn.functional.cross_entropy(logits[0],targets,reduction='none')

def token_objective(current,old,reference,advantage,mask,*,clip=.2,beta=.02,denominator=None):
    import torch
    old=old.to(current.device);reference=reference.to(current.device);mask=mask.to(current.device)
    if current.shape!=old.shape or current.shape!=reference.shape or current.shape!=mask.shape:raise ValueError('aligned action token evidence required')
    ratio=torch.exp(current-old);clipped=torch.clamp(ratio,1-clip,1+clip)
    delta=reference-current;kl=torch.exp(delta)-delta-1
    value=-torch.minimum(ratio*advantage,clipped*advantage)+beta*kl
    divisor=denominator if denominator is not None else mask.sum().clamp_min(1)
    loss=(value*mask).sum()/divisor
    if not torch.isfinite(loss.detach()).item():raise FloatingPointError('nonfinite GRPO objective')
    return loss


def corrected_token_objective(current,old_train,behavior,reference,advantage,mask,*,clip=.2,beta=.02,denominator=None,importance_cap=2.):
    """Token TIS for numeric rollout/training mismatch; old policy stays detached.

    PPO's ratio compares two training-path policies. The separate capped p_old/q
    factor corrects sampled token weighting; it is not proof of exact trajectory
    importance correction through all environment states.
    """
    import torch
    values=[x.detach().to(current.device) for x in (old_train,behavior,reference,mask)]
    old_train,behavior,reference,mask=values
    if any(x.shape!=current.shape for x in values):raise ValueError('aligned actual action token evidence required')
    if importance_cap<=0 or any(not torch.isfinite(x).all().item() for x in [current,*values]):
        raise FloatingPointError('nonfinite policy evidence or invalid importance cap')
    weights=torch.exp((old_train-behavior).clamp(max=__import__('math').log(importance_cap)))
    ratio=torch.exp(current-old_train);clipped=ratio.clamp(1-clip,1+clip)
    delta=reference-current;kl=torch.exp(delta)-delta-1
    value=-weights*torch.minimum(ratio*advantage,clipped*advantage)+beta*kl
    divisor=denominator if denominator is not None else mask.sum().clamp_min(1)
    loss=(value*mask).sum()/divisor
    if not torch.isfinite(loss.detach()).item():raise FloatingPointError('nonfinite corrected GRPO objective')
    return loss
