"""Same fixed20 runner, new prompt module only; no output/metric replacement."""
from pathlib import Path
ROOT = Path(__file__).resolve().parent
path = ROOT.parent / 'finance_program_skill_recovery20_20261021_r5/run_smoke.py'
namespace = dict(__file__=str(ROOT / 'run_smoke.py'), __name__='new_scope_same_fixed20_runner')
exec(compile(path.read_text(encoding='utf-8'), str(path)+'[new-scope-root]', 'exec'), namespace)
prepare, preflight, main = namespace['prepare'], namespace['preflight'], namespace['main']
previous = namespace['previous']
if __name__ == '__main__':
    import sys, json
    if sys.argv[1:] == ['--preflight']:
        print(json.dumps(preflight()))
    else:
        raise SystemExit(main())
