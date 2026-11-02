"""Use the proven native supervisor namespace directly, no nested entry lookup."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
path=ROOT.parent/'finance_program_skill_recovery20_20261021_r5/launch_smoke.py'
wrapper=dict(__file__=str(ROOT/'launch_smoke.py'),__name__='same_pair_supervisor_new_root')
exec(compile(path.read_text(encoding='utf-8'),str(path)+'[r9-root-only]','exec'),wrapper)
native=wrapper['namespace']
old_verify=wrapper['verify']

def verify():
    old_verify()
    stage=json.loads((ROOT/'stage.json').read_bytes())
    if stage['gpu_deadline_unix']!=1792621816.4901865 or stage['version_number']!=2:
        raise ValueError('campaign clock/version differs')
    old=ROOT.parent/'finance_program_skill_recovery20_20261022_r8'
    for rel in ('launch/exit.json','supervision/worker-exit.json'):
        if json.loads((old/rel).read_bytes())['actual_returncode']!=0:
            raise ValueError('R8 has not settled')
    for rel,keys in [('launch/started.json',['watcher_pid','supervisor_pid']),
                     ('supervision/worker-start.json',['pid']),('runs/started.json',['pid'])]:
        for key in keys:
            pid=json.loads((old/rel).read_bytes())[key]
            cmd=Path('/proc')/str(pid)/'cmdline'
            if cmd.exists() and str(old).encode() in cmd.read_bytes():
                raise ValueError('R8 writer still active')

def put(path,value):
    if Path(path)==ROOT/'admission.json':
        value.update(evaluation_stage='skill-recovery20-contract-projection-r9',
            training_gpu_deadline_unix=1792575276.1708825,
            context_window_tokens=49152,max_sequence_tokens=49152,training_updates=0)
    return wrapper['original_put'](path,value)

native.update(verify=verify,put=put)
if __name__=='__main__':
    raise SystemExit(native['watch']() if sys.argv[1:]==['--watch'] else native['main']())
