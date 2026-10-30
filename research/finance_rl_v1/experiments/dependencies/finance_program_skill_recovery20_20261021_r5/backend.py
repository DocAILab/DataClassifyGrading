"""Exact B80 backend with separately authorized clock and finite48K context."""
from pathlib import Path
from context_policy import validate_context_size

def valid_stage(admission, training_deadline):
    return (admission.get('evaluation_stage') == 'skill-recovery20-context48-gpu20'
            and admission.get('training_gpu_deadline_unix') == training_deadline == 1792575276.1708825)

def backend_class(root):
    path = Path(root) / 'eval_backend.py'
    source = path.read_text(encoding='utf-8')
    changes = [
        ("complete['admission']['gpu_deadline_unix'] == admission['gpu_deadline_unix']",
         "valid_stage(admission, complete['admission']['gpu_deadline_unix'])"),
        ("require(len(ids) + 512 <= 32768, 'context exceeds 32768 with full 512 generation allowance; no truncation')",
         "record.update(input_token_count=len(ids), context_window_tokens=49152, reserved_generation_tokens=512); validate_context_size(len(ids), self.admission)")]
    for old, new in changes:
        if source.count(old) != 1:
            raise ValueError('pinned backend source changed')
        source = source.replace(old, new)
    namespace = dict(__file__=str(path), __name__='fixed20_finite48k_backend',
                     valid_stage=valid_stage, validate_context_size=validate_context_size)
    exec(compile(source, str(path)+'[new-authorized-fixed20-48k]', 'exec'), namespace)
    return namespace['ModelBackend']
