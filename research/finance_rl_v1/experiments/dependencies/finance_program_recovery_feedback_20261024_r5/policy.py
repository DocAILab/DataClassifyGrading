import importlib.util
from pathlib import Path
path=Path(__file__).resolve().parent.parent/'finance_program_recovery_feedback_20261024_r4/policy.py'
spec=importlib.util.spec_from_file_location('same_fixed_gate_policy',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
gate_slots,gate_pass,pending_slots=module.gate_slots,module.gate_pass,module.pending_slots
