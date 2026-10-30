import importlib.util
from pathlib import Path
path=Path(__file__).resolve().parent.parent/'finance_program_evidence_consumption_20261024_r10/runtime.py'
spec=importlib.util.spec_from_file_location('n100_exact_runtime',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
build=module.build
previous=module.previous
