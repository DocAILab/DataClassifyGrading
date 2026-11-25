import importlib.util
from pathlib import Path
p=Path(__file__).resolve().parent.parent/'finance_program_evidence_consumption_20261024_r10/scope_binding.py'
s=importlib.util.spec_from_file_location('v13_exact_scope_binding',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
install=m.install
