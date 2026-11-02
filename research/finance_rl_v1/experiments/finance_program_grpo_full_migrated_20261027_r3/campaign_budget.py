def remaining(policy,attempts,*,now):
 if len(attempts)>=policy['maximum_attempts']:raise ValueError('attempt limit; no blind retry')
 if any(a['gpu_seconds']<0 or a['claims']<0 for a in attempts):raise ValueError('invalid usage')
 seconds=min(policy['training_seconds']-sum(a['gpu_seconds'] for a in attempts),policy['absolute_end_unix']-now)
 calls=policy['training_calls']-sum(a['claims'] for a in attempts)
 if seconds<=30 or calls<=0:raise ValueError('absolute/cumulative budget exhausted')
 return dict(seconds=seconds,calls=calls)
