import importlib.util
from pathlib import Path
path = Path(__file__).resolve().parent.parent / 'finance_program_skill_context_contract_20261021_r1/context_policy.py'
spec = importlib.util.spec_from_file_location('unchanged_finite_context', path)
previous = importlib.util.module_from_spec(spec); spec.loader.exec_module(previous)
validate_context_size = previous.validate_context_size
