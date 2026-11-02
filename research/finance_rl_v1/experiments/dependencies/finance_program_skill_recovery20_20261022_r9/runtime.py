import importlib.util
from pathlib import Path
path=Path(__file__).resolve().parent.parent/'finance_program_skill_recovery20_20261021_r5/runtime.py'
spec=importlib.util.spec_from_file_location('same_native_guards_and_budget',path)
previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
build,limits_remaining=previous.build,previous.limits_remaining
