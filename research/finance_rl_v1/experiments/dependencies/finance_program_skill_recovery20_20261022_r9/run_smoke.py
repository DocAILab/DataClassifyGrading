"""Exactly the settled R5 fixed20 runner, with the new root's prompt import."""
from pathlib import Path
ROOT=Path(__file__).resolve().parent
path=ROOT.parent/'finance_program_skill_recovery20_20261021_r5/run_smoke.py'
namespace=dict(__file__=str(ROOT/'run_smoke.py'),__name__='same_fixed20_contract_projection_runner')
exec(compile(path.read_text(encoding='utf-8'),str(path)+'[r9-root-only]','exec'),namespace)
prepare,preflight,main=(namespace[k] for k in ('prepare','preflight','main'))
previous=namespace['previous']
if __name__=='__main__':
    import sys,json
    if sys.argv[1:]==['--preflight']:print(json.dumps(preflight()))
    else:raise SystemExit(main())
