import importlib.util
from pathlib import Path
path=Path(__file__).resolve().parent.parent/'finance_program_skill_recovery20_20261021_r5/context_policy.py'
spec=importlib.util.spec_from_file_location('same_48k_context_policy',path)
previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
validate_context_size=previous.validate_context_size
