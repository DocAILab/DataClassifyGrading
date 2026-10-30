import importlib.util
from pathlib import Path
path = Path(__file__).resolve().parent.parent / 'finance_program_skill_recovery20_20261021_r5/resource_profile.py'
spec = importlib.util.spec_from_file_location('unchanged_pair_resource', path)
previous = importlib.util.module_from_spec(spec); spec.loader.exec_module(previous)
candidates, recheck = previous.candidates, previous.recheck
