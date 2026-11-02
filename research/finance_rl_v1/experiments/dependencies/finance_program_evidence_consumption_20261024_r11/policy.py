import importlib.util
from pathlib import Path
path=Path(__file__).resolve().parent.parent/'finance_program_evidence_consumption_20261024_r10/policy.py'
spec=importlib.util.spec_from_file_location('r11_exact_r10_slots',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
TARGETS,targeted,pending=module.TARGETS,module.targeted,module.pending
