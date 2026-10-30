"""Reuse exact finite48K no-truncation policy, no model configuration edit."""
import importlib.util
from pathlib import Path
path = Path(__file__).resolve().parent.parent / 'finance_program_skill_context_contract_20261021_r1/context_policy.py'
spec = importlib.util.spec_from_file_location('pinned_48k_context_policy', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
validate_context_size = module.validate_context_size
