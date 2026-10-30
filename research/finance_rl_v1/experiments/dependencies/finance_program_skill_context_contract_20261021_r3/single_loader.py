"""GPU0 text loader with explicit PEFT map; no automatic CPU language offload."""
from pathlib import Path
from single_policy import validate_admission
def device_map():return {'model.language_model':0,'lm_head':0,'model.visual':'cpu'}
def adapter_device_map():return {'base_model.model.'+k:v for k,v in device_map().items()}
def validate_placement(actual):
 if actual!=device_map():raise ValueError('only language/head on GPU0 and unused visual CPU permitted; no disk')
def validate_adapter_placement(model):
 from training_contract import ORIGINAL_LORA_NAMES
 if model.hf_device_map!=adapter_device_map():raise ValueError('PEFT must preserve exact explicit textGPU0/visualCPU map')
 actual={n:p for n,p in model.named_parameters() if 'lora_' in n}
 if set(actual)!=ORIGINAL_LORA_NAMES or any(p.device.type!='cuda' or p.device.index!=0 or p.is_meta for p in actual.values()):raise ValueError('all80 actual language LoRA tensors must be materialized on GPU0')
def adapt_source(source):
 changes=[('device_map="balanced"','device_map=device_map()'),
 ('if placements != {str(i) for i in range(count)}:\n        raise ProtocolError("model must occupy all admitted GPUs, with no CPU/disk placement")','validate_placement(base.hf_device_map)'),
 ('model = peft.PeftModel.from_pretrained(base, str(admission["original_adapter_dir"]), is_trainable=True, local_files_only=True)',
 'model = peft.PeftModel.from_pretrained(base, str(admission["original_adapter_dir"]), is_trainable=True, local_files_only=True, device_map=adapter_device_map(), max_memory={0:admission["model_max_memory_bytes"][0], "cpu":4*1024**3})\n    validate_adapter_placement(model)')]
 for old,new in changes:
  if source.count(old)!=1:raise ValueError('unknown pinned loader placement guard')
  source=source.replace(old,new)
 return source
def runtime_loader(loader,root,admission,tokenizer):
 validate_admission(admission)
 available=int(next(v.split()[1] for v in Path('/proc/meminfo').read_text().splitlines() if v.startswith('MemAvailable:')))*1024
 if available<4*1024**3:raise ValueError('insufficient CPU headroom for unused vision storage')
 namespace=dict(loader.__dict__,__name__='gpu0_inference_loader_only',device_map=device_map,validate_placement=validate_placement,adapter_device_map=adapter_device_map,validate_adapter_placement=validate_adapter_placement)
 exec(compile(adapt_source(Path(loader.__file__).read_text(encoding='utf-8')),str(loader.__file__)+'[explicit-base-and-PEFT-gpu0-maps]','exec'),namespace)
 return namespace['_runtime_loader'](root,admission,tokenizer)
