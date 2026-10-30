"""One isolated continuation, original resource admission and strict prior-exit gate."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('same_shared_launcher',ROOT.parent/'finance_program_skill_interface_smoke_20261021_r1/launch_smoke.py')
launcher=importlib.util.module_from_spec(s);s.loader.exec_module(launcher);launcher.ROOT=ROOT
launcher.DEADLINE=json.loads((ROOT/'stage.json').read_bytes())['gpu_deadline_unix']
old_verify=launcher.verify;old_put=launcher.put

def verify():
    old_verify()
    f=json.loads((ROOT/'freeze.json').read_bytes())
    for r in f['dependency_files']:
        if hashlib.sha256((ROOT.parent/r['path']).read_bytes()).hexdigest()!=r['sha256']:
            raise ValueError('pinned original dependency changed: '+r['path'])
    old=ROOT.parent/'finance_program_skill_recovery20_20261021_r1'
    for name in ['launch/exit.json','supervision/worker-exit.json']:
        if json.loads((old/name).read_bytes())['actual_returncode']!=0:raise ValueError('prior fixed20 did not exit cleanly')
    for name,keys in [('launch/started.json',['watcher_pid','supervisor_pid']),('supervision/worker-start.json',['pid']),('runs/started.json',['pid'])]:
        for key in keys:
            p=Path('/proc')/str(json.loads((old/name).read_bytes())[key])/'stat'
            if p.exists() and p.read_text().split(') ',1)[1].split()[0]!='Z':raise ValueError('prior writer remains live')

def put(path,value):
    if Path(path)==ROOT/'admission.json':
        value.update(evaluation_stage='skill-recovery20',training_gpu_deadline_unix=1792575276.1708825)
    return old_put(path,value)

launcher.verify=verify;launcher.put=put
if __name__=='__main__':raise SystemExit(launcher.watch() if sys.argv[1:]==['--watch'] else launcher.main())
