"""Reuse original shared-lock/timeout launcher; finite48K, one90min clock."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from resource_profile import candidates,recheck
s=importlib.util.spec_from_file_location('same_shared_launcher',ROOT.parent/'finance_program_skill_interface_smoke_20261021_r1/launch_smoke.py')
launcher=importlib.util.module_from_spec(s);s.loader.exec_module(launcher);launcher.ROOT=ROOT
launcher.DEADLINE=json.loads((ROOT/'stage.json').read_bytes())['gpu_deadline_unix']
original_verify=launcher.verify;original_put=launcher.put

def verify():
    original_verify()
    freeze=json.loads((ROOT/'freeze.json').read_bytes())
    for r in freeze['dependency_files']:
        if hashlib.sha256((ROOT.parent/r['path']).read_bytes()).hexdigest()!=r['sha256']:raise ValueError('original dependency changed: '+r['path'])
    old=ROOT.parent/'finance_program_skill_recovery20_20261021_r4'
    for name in ['launch/exit.json','supervision/worker-exit.json']:
        if json.loads((old/name).read_bytes())['actual_returncode']!=0:raise ValueError('old fixed20 not stopped cleanly')
    for name,keys in [('launch/started.json',['watcher_pid','supervisor_pid']),('supervision/worker-start.json',['pid']),('runs/started.json',['pid'])]:
        for key in keys:
            stat=Path('/proc')/str(json.loads((old/name).read_bytes())[key])/'stat'
            if stat.exists() and stat.read_text().split(') ',1)[1].split()[0]!='Z':raise ValueError('old writer live')

def put(path,value):
    if Path(path)==ROOT/'admission.json':value.update(evaluation_stage='context-contract-failure-replay',training_gpu_deadline_unix=1792575276.1708825,context_window_tokens=49152,max_sequence_tokens=49152)
    return original_put(path,value)

launcher.verify=verify;launcher.put=put
launcher.candidates=lambda snapshot,max_sequence_tokens:candidates(snapshot,max_sequence_tokens=49152)
launcher.recheck=recheck
if __name__=='__main__':raise SystemExit(launcher.watch() if sys.argv[1:]==['--watch'] else launcher.main())
