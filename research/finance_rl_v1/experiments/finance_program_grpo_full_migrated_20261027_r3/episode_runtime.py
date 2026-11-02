"""One fixed64 exploration; at most two genuinely on-policy smoke updates."""
from collections import defaultdict,Counter
from copy import deepcopy
import json
import math
from pathlib import Path
import time
from support import ROOT,N100,put,digest,load,native,verify
from execution_contract import validate_group


def grouped(rows):
    result={}
    for row in rows:result.setdefault(row['source_id'],[]).append(row)
    return list(result.values())

def assess(rows):
    groups=grouped(rows);complete=[]
    for g in groups:
        if len(g)==4:validate_group(g);complete.append(g)
    mixed=sum(len({r['reward']['outcome'] for r in g})>1 for g in complete)
    exact=all(r['native_replay_exact'] for r in rows)
    pred=sum(r['legal_completion'] for r in rows)
    return dict(denominator=len(rows),fixed_target=64,groups=len(complete),mixed_outcome_groups=mixed,
                all_wrong_groups=sum(not any(r['reward']['outcome'] for r in g) for g in complete),
                predictions=pred,algorithm_completed=sum(r['algorithm_completed'] for r in rows),
                joint_correct=sum(r['reward']['outcome'] for r in rows),
                stop_reasons=dict(Counter(r['stop_reason'] for r in rows)),all_exact=exact,
                allow_updates=len(rows)==64 and len(complete)==16 and mixed>=4 and pred>=58 and exact)


def put_native(path,value):
    import os
    path=Path(path);path.parent.mkdir(exist_ok=True,parents=True)
    with path.open('xb') as f:f.write(native().raw_bytes(value)+b'\n');f.flush();os.fsync(f.fileno())


def episode(env,backend,case,folder,seed,sample_index,reference):
    from reward import outcome,combine
    old=native();folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    torch=backend.runtime['torch'];torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    interface=load('rl_exact_per_episode_interface',N100/'interface.py')
    bridge=old.ContextBackend(case,interface.RecoveryAdapter(backend),'N',folder,backend.admission)
    trace,audit=env.run(case,bridge);put_native(folder/'trace.json',trace);put(folder/'dataflow.json',audit)
    replay=old.CPUReplay(bridge.records);again,other=env.run(case,replay)
    exact=old.raw_bytes(trace)==old.raw_bytes(again) and replay.position==len(bridge.records) and digest(audit)==digest(other)
    if not exact or not bridge.target_exact:raise old.TechnicalFailure('instrumented native replay or actual initial target differs')
    correct=outcome(trace,reference,audit['native_guard_valid'])
    reward=combine(correct,audit['invalid_count'],audit['dependency_count'])
    row=dict(folder=folder.relative_to(ROOT).as_posix(),source_id=case['source_id'],family=case['family'],
        sample_index=sample_index,policy_version=backend.version,seed=seed,stochastic=backend.stochastic,
        algorithm_completed=bool(trace['terminated']),legal_completion=bool(trace['terminated'] and audit['native_guard_valid']),
        prediction=trace['final_result'],stop_reason=trace['stop_reason'],executed_steps=trace['new_steps'],
        model_calls=len(bridge.records),planner_attempts=len(trace['planner_attempts']),reward=reward,
        native_replay_exact=exact,actual_initial_exact=bridge.target_exact,
        dependency_training_eligible=audit['dependency_training_eligible'])
    put(folder/'result.json',row)
    print(json.dumps({k:v for k,v in row.items() if k not in ('prediction','family','reward')},ensure_ascii=True),flush=True)
    return row


def main():
    verify()
    from native_env import Environment
    from backend import make_backend,planner_turns
    from policy_update import update
    public=json.loads((ROOT/'cohort-public.json').read_bytes());labels=json.loads((ROOT/'reward-labels.json').read_bytes())
    admission=json.loads((ROOT/'admission.json').read_bytes());env=Environment(public)
    run=ROOT/'runs';run.mkdir(exist_ok=False);rows=[];updates=[];post=[]
    put(run/'started.json',dict(time_unix=time.time(),source_denominator=16,rollouts=64,maximum_updates=2,private_labels_not_in_public=True))
    backend=make_backend(admission)
    try:
        backend._ensure_loaded();put(run/'initial-model-identity.json',backend.identity)
        for index,case in enumerate(env.cases):
            for sample in range(4):
                row=episode(env,backend,case,run/f'explore/{index+1:02d}-{sample+1}',20261025+index*101+sample,sample,labels[case['source_id']]);rows.append(row)
                if len(rows)==1:
                    # Real same-backend probability check before paying for64 episodes.
                    first=planner_turns(ROOT/row['folder'])[0]
                    actual=backend.log_probs(first);old=backend.runtime['torch'].tensor(first['old_log_probs'])
                    delta=(actual-old).abs();proof=dict(max_abs=float(delta.max()),mean_abs=float(delta.mean()),turn=first['path'])
                    put(run/'first-probability-check.json',proof)
                    if proof['max_abs']>.15 or proof['mean_abs']>.025:raise ValueError('behavior/recomputed log-prob mismatch')
        gate=assess(rows);put(run/'exploration.json',dict(gate,rows=rows,time_unix=time.time()))
        if gate['allow_updates']:
            selected=[g for g in grouped(rows) if len({r['reward']['outcome'] for r in g})>1][:4]
            reward_name='A4' if all(r['dependency_training_eligible'] for g in selected for r in g) else 'A2'
            put(run/'reward-admission.json',dict(selected_reward=reward_name,
                reason='complete_dependency_audit' if reward_name=='A4' else 'explicit_bindings_audited_but_implicit_skill_prompt_augmentation_not_fully_proven',
                no_fabricated_dependency_credit=True))
            updates.append(update(backend,selected,reward_name))
            if updates[-1]['updated']:
                # A second optimizer batch must be regenerated under version1.
                next_groups=[]
                by={c['source_id']:c for c in env.cases}
                for index,g in enumerate(selected):
                    case=by[g[0]['source_id']];fresh=[]
                    for sample in range(4):fresh.append(episode(env,backend,case,run/f'update2/{index+1:02d}-{sample+1}',20262025+index*101+sample,sample,labels[case['source_id']]))
                    if len({r['reward']['outcome'] for r in fresh})>1:next_groups.append(fresh)
                if next_groups:updates.append(update(backend,next_groups,reward_name))
                backend.stochastic=False
                for index,case in enumerate(env.cases):post.append(episode(env,backend,case,run/f'greedy-post/{index+1:02d}',20263025+index,0,labels[case['source_id']]))
        result=dict(completed=True,kind='bounded_training_source_GRPO_pilot',new_exploration_episodes=64,exploration=gate,
            updates=updates,actual_optimizer_updates=backend.version,post_update_greedy=post,
            no_update_reason=None if backend.version else ('exploration_gate_not_met' if not gate['allow_updates'] else 'no_nonzero_advantages'),
            total_claims=len(list((ROOT/'campaign/claims').glob('*.json'))),maximum_claims=4000,
            reference_digest=backend.reference_digest,actor_tensor_digest=digest(backend.weight_shas),
            full_RL_ablation_complete=False,fresh_test=False,time_unix=time.time())
        verify();put(run/'completion.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ('updates','post_update_greedy')},ensure_ascii=True),flush=True)
        return 0
    except BaseException as exc:
        put(run/'failure.json',dict(error_type=type(exc).__name__,error=str(exc),completed_exploration=len(rows),rows=rows,
            updates=updates,actual_optimizer_updates=backend.version,post=post,time_unix=time.time()))
        raise


if __name__=='__main__':
    from continuation import run
    raise SystemExit(run())
