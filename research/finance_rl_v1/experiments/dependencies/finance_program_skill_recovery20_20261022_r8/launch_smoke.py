"""Existing shared-lock/timeout supervision with a new finite campaign clock."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
path = ROOT.parent / 'finance_program_skill_recovery20_20261021_r5/launch_smoke.py'
namespace = dict(__file__=str(ROOT / 'launch_smoke.py'), __name__='new_scope_same_safe_supervisor')
exec(compile(path.read_text(encoding='utf-8'), str(path)+'[new-scope-root]', 'exec'), namespace)
old_verify = namespace['verify']

def verify():
    old_verify()
    for name in ('finance_program_skill_recovery20_20261021_r5', 'finance_program_skill_recovery20_20261021_r7'):
        old = ROOT.parent / name
        for rel in ('launch/exit.json', 'supervision/worker-exit.json'):
            if json.loads((old / rel).read_bytes())['actual_returncode'] != 0:
                raise ValueError('prior GPU stage has no successful settled exit')
        for rel, keys in [('launch/started.json', ['watcher_pid', 'supervisor_pid']),
                          ('supervision/worker-start.json', ['pid']), ('runs/started.json', ['pid'])]:
            for key in keys:
                pid = json.loads((old / rel).read_bytes())[key]
                cmd = Path('/proc') / str(pid) / 'cmdline'
                if cmd.exists() and str(old).encode() in cmd.read_bytes():
                    raise ValueError('prior GPU writer still has its actual command identity')

def put(path, value):
    if Path(path) == ROOT / 'admission.json':
        value.update(evaluation_stage='skill-recovery20-taskscope-r8',
                     training_gpu_deadline_unix=1792575276.1708825,
                     context_window_tokens=49152, max_sequence_tokens=49152, training_updates=0)
    return namespace['original_put'](path, value)

namespace.update(verify=verify, put=put)
if __name__ == '__main__':
    raise SystemExit(namespace['watch']() if sys.argv[1:] == ['--watch'] else namespace['main']())
