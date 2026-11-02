"""One real on-policy optimizer update; whole episode advantages, action tokens."""
import math
import time
from pathlib import Path
from backend import planner_turns
from grpo_math import advantages
from execution_contract import validate_group
from support import ROOT,put,digest


def update(backend,groups,reward_name='A2',*,turn_loader=None):
    from policy_math import corrected_token_objective
    from checkpoint_store import save,restore
    from full_contract import ensure_update_allowed
    ensure_update_allowed(backend)
    torch=backend.runtime['torch'];model=backend.runtime['model']
    for group in groups:
        validate_group(group)
        if any(r['policy_version']!=backend.version for r in group):raise ValueError('stale rollout batch')
    adv=advantages([[r['reward'][reward_name] for r in g] for g in groups],[[r['reward']['outcome'] for r in g] for g in groups])
    turns=[]
    for group,aa in zip(groups,adv['advantages']):
        for row,a in zip(group,aa):
            if not a:continue
            for turn in (turn_loader(row) if turn_loader else planner_turns(ROOT/row['folder'])):turns.append((turn,a))
    if not turns:return dict(updated=False,reason='no_nonzero_valid_group_advantage',advantages=adv)
    if not getattr(backend,'optimizer',None):backend.optimizer=torch.optim.AdamW([p for _,p in backend.named],lr=5e-6,betas=(.9,.999),eps=1e-8,weight_decay=0.)
    optimizer=backend.optimizer;optimizer.zero_grad(set_to_none=True)
    before={n:p.detach().cpu().clone() for n,p in backend.named};token_count=sum(len(t['tokens']) for t,_ in turns)
    step=backend.version+1;folder=ROOT/f'runs/updates/step-{step:02d}';folder.mkdir(parents=True,exist_ok=False)
    put(folder/'started.json',dict(policy_version=backend.version,reward=reward_name,groups=[g[0]['source_id'] for g in groups],
        planner_turns=len(turns),action_tokens=token_count,advantages=adv,time_unix=time.time()))
    micro=[]
    from host_cache import trim_host_cache
    try:
        for index,(turn,a) in enumerate(turns,1):
            behavior=torch.tensor(turn['old_log_probs'],dtype=torch.float32)
            reference=backend.log_probs(turn,reference=True)
            old=backend.log_probs(turn)
            with torch.autograd.graph.save_on_cpu(pin_memory=True):
                current=backend.log_probs(turn,grad=True)
                delta=(current.detach().cpu()-old).abs()
                mismatch=(old-behavior).abs();weights=torch.exp((old-behavior).clamp(max=math.log(2.)))
                check=dict(max_abs_logp_delta=float(delta.max()),mean_abs_logp_delta=float(delta.mean()),
                    behavior_max_abs_delta=float(mismatch.max()),behavior_mean_abs_delta=float(mismatch.mean()),
                    importance_min=float(weights.min()),importance_max=float(weights.max()),importance_mean=float(weights.mean()),
                    importance_cap_fraction=float(((old-behavior)>math.log(2.)).float().mean()),
                    importance_relative_ESS=float(weights.sum().square()/(weights.square().sum()*len(weights))),
                    old_training_policy_sha256=digest(old.tolist()),behavior_sha256=digest(behavior.tolist()))
                put(folder/f'micro-{index:03d}.forward.json',dict(turn=turn['path'],advantage=a,**check))
                if check['max_abs_logp_delta']>.01 or check['mean_abs_logp_delta']>.001:
                    raise ValueError('same training-path old/current actor mismatch before optimizer step')
                if not all(math.isfinite(check[k]) for k in ('importance_min','importance_max','importance_relative_ESS')) or check['importance_relative_ESS']<.9 or check['importance_cap_fraction']>.05:
                    raise ValueError('rollout/training TIS correction outside admitted ESS/cap bounds')
                loss=corrected_token_objective(current,old,behavior,reference,a,torch.ones_like(current),denominator=token_count)
                loss_value=float(loss.detach());loss.backward()
                micro.append(dict(turn=turn['path'],advantage=a,loss=loss_value,**check))
                put(folder/f'micro-{index:03d}.backward.json',micro[-1])
            del current,loss,reference,old,behavior,weights
            trim_host_cache(torch)
        from deps import math_runtime
        norm,clipped=math_runtime.global_clip(backend.named,torch)
        backend.resources()
        put(folder/'optimizer-start.json',dict(time_unix=time.time(),grad_norm=norm))
        optimizer.step()
        for _,p in backend.named:
            state=optimizer.state[p]
            if float(state['step'].item())!=step:raise ValueError('optimizer step count differs')
            if any(torch.is_tensor(v) and (v.dtype!=torch.float32 or not torch.isfinite(v).all().item()) for v in state.values()):
                raise ValueError('optimizer state must be finite FP32')
        change=math.sqrt(sum(float((p.detach().cpu()-before[n]).double().square().sum()) for n,p in backend.named))
        if not math.isfinite(change) or change<=0:raise ValueError('actual optimizer update must be finite and nonzero')
        for _,p in backend.named:
            if not torch.isfinite(p).all().item():raise FloatingPointError('nonfinite updated actor')
        reference_now={n:__import__('hashlib').sha256(v.contiguous().numpy().tobytes()).hexdigest() for n,v in backend.reference.items()}
        if digest(reference_now)!=backend.reference_digest:raise ValueError('frozen reference mutated')
        backend.version+=1;backend.freeze_for_generation()
        checkpoint=save(ROOT/f'runs/checkpoints/version-{backend.version:02d}',backend.named,optimizer,version=backend.version,reference_digest=backend.reference_digest)
        # Restore the new complete boundary and prove exact round-trip, without
        # re-running rollout or changing the optimizer step count.
        expected=backend.tensor_hashes();restore(ROOT/f'runs/checkpoints/version-{backend.version:02d}',backend.named,optimizer,reference_digest=backend.reference_digest)
        if backend.tensor_hashes()!=expected:raise ValueError('real checkpoint roundtrip differs')
        backend.freeze_for_generation()
        result=dict(updated=True,policy_version=backend.version,reward=reward_name,grad_norm=norm,clipped_grad_norm=clipped,
                    update_norm=change,action_tokens=token_count,planner_turns=len(turns),checkpoint=checkpoint,
                    reference_unchanged=True,checkpoint_roundtrip_exact=True,micro=micro,time_unix=time.time())
        put(folder/'completion.json',result);return result
    except BaseException as exc:
        put(folder/'failure.json',dict(error_type=type(exc).__name__,error=str(exc),policy_version=backend.version,
            completed_backward_micro=len(micro),time_unix=time.time()))
        raise
