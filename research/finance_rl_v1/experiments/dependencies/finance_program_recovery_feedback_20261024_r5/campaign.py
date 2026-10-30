"""Resume the same bounded round; never synthesize a fresh deadline."""
def resume_admission(original,combo,*,now,campaign_root):
    if now>=original['gpu_deadline_unix']-600:raise TimeoutError('original round has insufficient remaining window')
    value=dict(original);value.update(combo)
    value.update(evaluation_only=True,campaign_root=campaign_root,resume_admitted_unix=now)
    value['gpu_started_unix']=original['gpu_started_unix'];value['gpu_deadline_unix']=original['gpu_deadline_unix']
    return value
