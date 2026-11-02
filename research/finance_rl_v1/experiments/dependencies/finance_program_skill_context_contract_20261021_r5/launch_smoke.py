"""Two locked inference GPUs, original cumulative clock and native backend."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from resource_profile import candidates,recheck
OLD=ROOT.parent/'finance_program_skill_context_contract_20261021_r1'
PREVIOUS=ROOT.parent/'finance_program_skill_context_contract_20261021_r3'
path=ROOT.parent/'finance_program_skill_interface_smoke_20261021_r1/launch_smoke.py'
source=path.read_text(encoding='utf-8')
old="if len(combo['gpu_indices']) != 3:"
if source.count(old)!=1:raise ValueError('unknown original GPU count guard')
namespace={'__file__':str(path),'__name__':'two_gpu_same_supervisor'}
exec(compile(source.replace(old,"if len(combo['gpu_indices']) != 2:"),str(path)+'[two-inference-cards]','exec'),namespace)
s=importlib.util.spec_from_file_location('original_context_freeze_verify',OLD/'launch_smoke.py')
original=importlib.util.module_from_spec(s);s.loader.exec_module(original)
put_original=namespace['put']
def verify():
 original.verify()
 failed=ROOT.parent/'finance_program_skill_context_contract_20261021_r4'
 for name in ['launch/exit.json','supervision/worker-exit.json']:
  if json.loads((failed/name).read_bytes())['actual_returncode']!=1:raise ValueError('previous twoGPU failed tree not settled')
 for pid in [2927201,2927235,2927268,2927269]:
  p=Path('/proc')/str(pid)/'stat'
  if p.exists() and p.read_text().split(') ',1)[1].split()[0]!='Z':raise ValueError('previous twoGPU writer still live')
 freeze=json.loads((ROOT/'freeze.json').read_bytes())
 for row in freeze['files']:
  p=ROOT/row['path']
  if p.is_symlink() or p.stat().st_size!=row['bytes'] or hashlib.sha256(p.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('two GPU freeze changed')
 for row in freeze['dependency_files']:
  if hashlib.sha256((ROOT.parent/row['path']).read_bytes()).hexdigest()!=row['sha256']:raise ValueError('frozen dependency changed: '+row['path'])
 for name in ['launch/exit.json','supervision/worker-exit.json']:
  if json.loads((PREVIOUS/name).read_bytes())['actual_returncode']!=1:raise ValueError('singleGPU OOM actualexit missing')
 for pid in [2921177,2921209,2921260,2921261]:
  p=Path('/proc')/str(pid)/'stat'
  if p.exists() and p.read_text().split(') ',1)[1].split()[0]!='Z':raise ValueError('singleGPU writer stilllive')
 for key in ['input-13','input-14']:
  result=json.loads((PREVIOUS/'runs'/key/'result.json').read_bytes())
  if not result['native_valid'] or not result['generation_success']:raise ValueError('reused Skill evidence not valid')
def put(path,value):
 if Path(path)==ROOT/'admission.json':value.update(evaluation_stage='context-contract-failure-replay',training_gpu_deadline_unix=1792575276.1708825,context_window_tokens=49152,max_sequence_tokens=49152,training_updates=0)
 return put_original(path,value)
namespace.update(ROOT=ROOT,DEADLINE=1792587659.6128213,verify=verify,put=put,candidates=lambda rows,max_sequence_tokens:candidates(rows,max_sequence_tokens=49152),recheck=recheck)
if __name__=='__main__':raise SystemExit(namespace['watch']() if sys.argv[1:]==['--watch'] else namespace['main']())
