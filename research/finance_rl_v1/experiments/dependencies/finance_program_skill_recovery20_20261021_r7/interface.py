import importlib.util
from pathlib import Path
path = Path(__file__).resolve().parent.parent / 'finance_program_skill_recovery20_20261021_r6/interface.py'
spec = importlib.util.spec_from_file_location('unchanged_synthesis_prompt_v3', path)
previous = importlib.util.module_from_spec(spec); spec.loader.exec_module(previous)
rewrite = previous.rewrite
