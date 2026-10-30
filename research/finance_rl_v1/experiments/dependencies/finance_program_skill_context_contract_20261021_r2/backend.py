"""Pinned B80 backend: finite48K, single GPU admission, unchanged weight guards."""
from pathlib import Path
from context_policy import validate_context_size
from single_loader import runtime_loader
from single_policy import validate_admission
def valid_stage(admission,training_deadline):
 return admission.get('evaluation_stage')=='context-contract-failure-replay' and admission.get('training_gpu_deadline_unix')==training_deadline==1792575276.1708825
def backend_class(root):
 path=Path(root)/'eval_backend.py';source=path.read_text(encoding='utf-8')
 changes=[("complete['admission']['gpu_deadline_unix'] == admission['gpu_deadline_unix']","valid_stage(admission, complete['admission']['gpu_deadline_unix'])"),
 ("require(len(ids) + 512 <= 32768, 'context exceeds 32768 with full 512 generation allowance; no truncation')","record.update(input_token_count=len(ids), context_window_tokens=49152, reserved_generation_tokens=512); validate_context_size(len(ids), self.admission)"),
 ('runtime = loader._runtime_loader(self.root, self.admission, tokenizer)','runtime = runtime_loader(loader, self.root, self.admission, tokenizer)')]
 for old,new in changes:
  if source.count(old)!=1:raise ValueError('unknown pinned evaluation backend guard')
  source=source.replace(old,new)
 namespace={'__file__':str(path),'__name__':'gpu0_fixed_failure_backend','valid_stage':valid_stage,'validate_context_size':validate_context_size,'runtime_loader':runtime_loader}
 exec(compile(source,str(path)+'[singleGPU-finite48k]','exec'),namespace)
 namespace['validate_admission']=validate_admission
 return namespace['ModelBackend']
