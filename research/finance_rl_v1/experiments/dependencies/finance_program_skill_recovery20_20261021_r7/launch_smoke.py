"""Same supervisor, also refuse any still-live initialization-failure writer."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parent
path = ROOT.parent / 'finance_program_skill_recovery20_20261021_r6/launch_smoke.py'
outer = dict(__file__=str(ROOT / 'launch_smoke.py'), __name__='single_skill_init_fixed_supervisor')
exec(compile(path.read_text(encoding='utf-8'), str(path)+'[settled-r6-required]', 'exec'), outer)
namespace = outer['namespace']
parent_verify = namespace['verify']

def verify():
    parent_verify()
    previous = ROOT.parent / 'finance_program_skill_recovery20_20261021_r6'
    for name in ('launch/exit.json', 'supervision/worker-exit.json'):
        assert json.loads((previous / name).read_bytes())['actual_returncode'] == 1
    for pid in (3058052, 3058071, 3058102, 3058103):
        stat = Path('/proc') / str(pid) / 'stat'
        assert not stat.exists() or stat.read_text().split(') ', 1)[1].split()[0] == 'Z'

namespace['verify'] = verify
if __name__ == '__main__':
    raise SystemExit(namespace['watch']() if sys.argv[1:] == ['--watch'] else namespace['main']())
