"""Pinned previous fixed20 runner, new prompt module and isolated destinations."""
import importlib.util
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
s=importlib.util.spec_from_file_location('pinned_fixed20_worker',ROOT.parent/'finance_program_skill_recovery20_20261021_r1/run_smoke.py')
worker=importlib.util.module_from_spec(s);s.loader.exec_module(worker);worker.ROOT=ROOT
prepare=worker.prepare
preflight=worker.preflight
CPUReplay=worker.CPUReplay
raw_bytes=worker.raw_bytes
rewrite=worker.rewrite
if __name__=='__main__':
    if sys.argv[1:]==['--preflight']:
        import json
        print(json.dumps(preflight()))
    else:raise SystemExit(worker.main())
