import importlib.util
from pathlib import Path
p=Path(__file__).resolve().parent.parent/'finance_program_evidence_consumption_20261024_r11/feedback.py'
s=importlib.util.spec_from_file_location('v13_exact_feedback',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
project=m.project
