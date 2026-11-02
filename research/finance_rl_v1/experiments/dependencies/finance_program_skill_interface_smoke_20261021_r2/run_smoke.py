"""Reuse the pinned r1 diagnostic, replace only its prompt adaptation layer."""
import importlib.util
from pathlib import Path
import sys
from next_contracts import rewrite, validate_output

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent / 'finance_program_skill_interface_smoke_20261021_r1'
sys.path.insert(0, str(PREVIOUS))
spec = importlib.util.spec_from_file_location('previous_smoke_worker', PREVIOUS / 'run_smoke.py')
worker = importlib.util.module_from_spec(spec); spec.loader.exec_module(worker)
worker.ROOT = ROOT
worker.rewrite = rewrite
worker.validate_output = validate_output

if __name__ == '__main__':
    raise SystemExit(worker.main())
