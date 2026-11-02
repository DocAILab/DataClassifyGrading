"""Same original BF16/80-FP32 loader; only inactive vision storage is CPU."""
from pathlib import Path
from single_policy import validate_admission
def device_map():return {'model.language_model':0,'lm_head':0,'model.visual':'cpu'}
def validate_placement(actual):
 if actual!=device_map():raise ValueError('only language/head on GPU0 and unused visual CPU permitted; no disk')
def adapt_source(source):
 changes=[('device_map="balanced"','device_map=device_map()'),
 ('if placements != {str(i) for i in range(count)}:\n        raise ProtocolError("model must occupy all admitted GPUs, with no CPU/disk placement")','validate_placement(base.hf_device_map)')]
 for old,new in changes:
  if source.count(old)!=1:raise ValueError('unknown pinned training loader placement guard')
  source=source.replace(old,new)
 return source
def runtime_loader(loader,root,admission,tokenizer):
 validate_admission(admission)
 available=int(next(v.split()[1] for v in Path('/proc/meminfo').read_text().splitlines() if v.startswith('MemAvailable:')))*1024
 if available<4*1024**3:raise ValueError('insufficient CPU headroom for unused vision storage')
 namespace=dict(loader.__dict__,__name__='gpu0_inference_loader_only',device_map=device_map,validate_placement=validate_placement)
 source=adapt_source(Path(loader.__file__).read_text(encoding='utf-8'))
 exec(compile(source,str(loader.__file__)+'[gpu0-visual-cpu-only]','exec'),namespace)
 return namespace['_runtime_loader'](root,admission,tokenizer)
