def limits(policy,spent,kind,*,now):
 if kind!='evaluation' or not 1<=spent.get('updates',0)<=6:raise ValueError('accepted real trained policy required')
 seconds=min(policy['evaluation_seconds'],policy['total_gpu_seconds']-spent['gpu_seconds'],policy['absolute_end_unix']-now)
 calls=min(policy['evaluation_calls'],policy['total_calls']-spent['calls'])
 if seconds<=60 or calls<=0:raise ValueError('campaign budget exhausted')
 return dict(gpu_deadline_unix=now+seconds,model_call_limit=int(calls),maximum_optimizer_updates=0,inherited_claims=0)
