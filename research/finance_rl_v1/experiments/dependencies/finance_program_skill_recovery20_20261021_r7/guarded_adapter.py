import importlib.util
from pathlib import Path
path = Path(__file__).resolve().parent.parent / 'finance_program_skill_recovery20_20261021_r6/guarded_adapter.py'
spec = importlib.util.spec_from_file_location('unchanged_malformed_guard_adapter', path)
previous = importlib.util.module_from_spec(spec); spec.loader.exec_module(previous)
GuardedAdapter = previous.GuardedAdapter
