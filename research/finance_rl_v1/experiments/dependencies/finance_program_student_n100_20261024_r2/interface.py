import importlib.util
from pathlib import Path
path=Path(__file__).resolve().parent.parent/'finance_program_evidence_consumption_20261024_r11/interface.py'
spec=importlib.util.spec_from_file_location('n100_exact_interface',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
rewrite=module.rewrite
RecoverySession=module.RecoverySession
RecoveryAdapter=module.RecoveryAdapter
