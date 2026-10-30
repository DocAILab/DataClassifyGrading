"""One targeted Skill check, reusing the parent's unchanged absolute deadline."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parent
path = ROOT.parent / 'finance_program_skill_recovery20_20261021_r5/launch_smoke.py'
namespace = dict(__file__=str(ROOT / 'launch_smoke.py'), __name__='single_synthesis_supervisor')
exec(compile(path.read_text(encoding='utf-8'), str(path)+'[one-target-no-clock-reset]', 'exec'), namespace)
verify_parent_dependencies = namespace['verify']
namespace = namespace['namespace']

def verify():
    verify_parent_dependencies()
    parent = ROOT.parent / 'finance_program_skill_recovery20_20261021_r5'
    for name in ('launch/exit.json', 'supervision/worker-exit.json'):
        assert json.loads((parent / name).read_bytes())['actual_returncode'] == 0
    for pid in (3025321, 3025341, 3025928, 3025929):
        stat = Path('/proc') / str(pid) / 'stat'
        assert not stat.exists() or stat.read_text().split(') ', 1)[1].split()[0] == 'Z'

namespace['verify'] = verify
if __name__ == '__main__':
    raise SystemExit(namespace['watch']() if sys.argv[1:] == ['--watch'] else namespace['main']())
