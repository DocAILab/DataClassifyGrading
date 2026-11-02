"""Separate authorized evaluation clock; preserve immutable training provenance."""
from pathlib import Path

def valid_stage(admission,training_deadline):
    return (admission.get('evaluation_stage')=='skill-recovery20'
            and admission.get('training_gpu_deadline_unix')==training_deadline==1792575276.1708825)

def backend_class(root):
    path=Path(root)/'eval_backend.py';source=path.read_text(encoding='utf-8')
    old="complete['admission']['gpu_deadline_unix'] == admission['gpu_deadline_unix']"
    if source.count(old)!=1:raise ValueError('unexpected pinned backend deadline guard')
    source=source.replace(old,"valid_stage(admission, complete['admission']['gpu_deadline_unix'])")
    namespace={'__file__':str(path),'__name__':'separate_clock_pinned_backend','valid_stage':valid_stage}
    exec(compile(source,str(path)+'[separate-authorized-evaluation-clock]','exec'),namespace)
    return namespace['ModelBackend']
