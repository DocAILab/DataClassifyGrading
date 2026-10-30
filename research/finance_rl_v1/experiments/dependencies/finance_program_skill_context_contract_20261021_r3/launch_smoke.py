"""Reuse actual inherited-lock/timeout supervision, now GPU0 only."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from single_policy import candidates,recheck,DEADLINE
OLD=ROOT.parent/'finance_program_skill_context_contract_20261021_r1'
source_path=ROOT.parent/'finance_program_skill_interface_smoke_20261021_r1/launch_smoke.py'
source=source_path.read_text(encoding='utf-8')
old="if len(combo['gpu_indices']) != 3:"
if source.count(old)!=1:raise ValueError('unknown pinned supervisor card-count guard')
source=source.replace(old,"if len(combo['gpu_indices']) != 1:")
namespace={'__file__':str(source_path),'__name__':'gpu0_original_supervision'}
exec(compile(source,str(source_path)+'[one-card-only]','exec'),namespace)
s=importlib.util.spec_from_file_location('stopped_context_r1_launcher',OLD/'launch_smoke.py')
original=importlib.util.module_from_spec(s);s.loader.exec_module(original)
put_original=namespace['put']
def verify():
 original.verify()
 previous=ROOT.parent/'finance_program_skill_context_contract_20261021_r2'
 for item in ['launch/exit.json','supervision/worker-exit.json']:
  if json.loads((previous/item).read_bytes())['actual_returncode']!=1:raise ValueError('previous failed loader actual exit missing')
 if not (previous/'runs/failure.json').exists() or (previous/'runs/loaded.json').exists() or list((previous/'runs').glob('*/model-request.json')):raise ValueError('previous pre-generation load failure boundary changed')
 for pid in [2916454,2916484,2916509,2916510]:
  p=Path('/proc')/str(pid)/'stat'
  if p.exists() and p.read_text().split(') ',1)[1].split()[0]!='Z':raise ValueError('previous GPU0 loader still live')
 for r in json.loads((ROOT/'freeze.json').read_bytes())['files']:
  p=ROOT/r['path']
  if p.is_symlink() or p.stat().st_size!=r['bytes'] or hashlib.sha256(p.read_bytes()).hexdigest()!=r['sha256']:raise ValueError('single GPU freeze changed: '+r['path'])
 for r in json.loads((ROOT/'freeze.json').read_bytes())['dependency_files']:
  if hashlib.sha256((ROOT.parent/r['path']).read_bytes()).hexdigest()!=r['sha256']:raise ValueError('pinned dependency changed: '+r['path'])
 if json.loads((OLD/'launch/exit.json').read_bytes())['actual_returncode']!=-15:raise ValueError('old idle waiter cancellation receipt missing')
 if (OLD/'runs').exists() or (OLD/'supervision/worker-start.json').exists():raise ValueError('old three GPU model worker existed; never repeat its inputs')
 for pid in [2899037,2899038]:
  p=Path('/proc')/str(pid)/'stat'
  if p.exists() and p.read_text().split(') ',1)[1].split()[0]!='Z':raise ValueError('old waiter remains live')
def put(path,value):
 if Path(path)==ROOT/'admission.json':value.update(evaluation_stage='context-contract-failure-replay',training_gpu_deadline_unix=1792575276.1708825,context_window_tokens=49152,max_sequence_tokens=49152,inference_placement='textGPU0-unusedVisionCPU',training_updates=0)
 return put_original(path,value)
namespace.update(ROOT=ROOT,DEADLINE=DEADLINE,verify=verify,put=put,candidates=lambda rows,max_sequence_tokens:candidates(rows,max_sequence_tokens=49152),recheck=recheck)
if __name__=='__main__':raise SystemExit(namespace['watch']() if sys.argv[1:]==['--watch'] else namespace['main']())
