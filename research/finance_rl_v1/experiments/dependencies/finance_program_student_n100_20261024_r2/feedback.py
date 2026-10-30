import importlib.util
from pathlib import Path
path=Path(__file__).resolve().parent.parent/'finance_program_evidence_consumption_20261024_r11/feedback.py'
spec=importlib.util.spec_from_file_location('n100_exact_feedback',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
project=module.project
