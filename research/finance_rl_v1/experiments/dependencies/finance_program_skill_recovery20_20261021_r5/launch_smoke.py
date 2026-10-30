"""Reuse owned timeout/process supervision; exact selected pair, new finite stage."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
E2 = ROOT.parent/'finance_program_student_b80_eval_20261021_r2'
sys.path.insert(0, str(ROOT))
from resource_profile import candidates, recheck
path = ROOT.parent/'finance_program_skill_interface_smoke_20261021_r1/launch_smoke.py'
source = path.read_text(encoding='utf-8')
old = "if len(combo['gpu_indices']) != 3:"
if source.count(old) != 1:
    raise ValueError('unknown original supervisor count guard')
namespace = dict(__file__=str(path), __name__='fixed20_selected_pair_supervisor')
exec(compile(source.replace(old, "if len(combo['gpu_indices']) != 2:"), str(path)+'[selected-pair]', 'exec'), namespace)
original_verify, original_put = namespace['verify'], namespace['put']
stage = json.loads((ROOT/'stage.json').read_bytes())

def verify():
    original_verify()
    freeze = json.loads((ROOT/'freeze.json').read_bytes())
    for row in freeze['dependency_files']:
        path = ROOT.parent/row['path']
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError('frozen dependency changed: '+row['path'])
    for name, expected in [('finance_program_skill_recovery20_20261021_r4', 0),
                           ('finance_program_skill_context_contract_20261021_r5', 1)]:
        old_root = ROOT.parent/name
        for rel in ('launch/exit.json', 'supervision/worker-exit.json'):
            if json.loads((old_root/rel).read_bytes())['actual_returncode'] != expected:
                raise ValueError('old GPU stage has no settled actual exit')
        for rel, keys in [('launch/started.json', ['watcher_pid', 'supervisor_pid']),
                          ('supervision/worker-start.json', ['pid']), ('runs/started.json', ['pid'])]:
            for key in keys:
                pid = json.loads((old_root/rel).read_bytes())[key]
                stat = Path('/proc')/str(pid)/'stat'
                if stat.exists() and stat.read_text().split(') ', 1)[1].split()[0] != 'Z':
                    raise ValueError('old GPU writer remains live')

def put(path, value):
    if Path(path) == ROOT/'admission.json':
        value.update(evaluation_stage='skill-recovery20-context48-gpu20',
                     training_gpu_deadline_unix=1792575276.1708825,
                     context_window_tokens=49152, max_sequence_tokens=49152, training_updates=0)
    return original_put(path, value)

namespace.update(ROOT=ROOT, DEADLINE=stage['gpu_deadline_unix'], verify=verify, put=put,
                 candidates=lambda rows, max_sequence_tokens: candidates(rows, max_sequence_tokens=49152), recheck=recheck)
if __name__ == '__main__':
    raise SystemExit(namespace['watch']() if sys.argv[1:] == ['--watch'] else namespace['main']())
