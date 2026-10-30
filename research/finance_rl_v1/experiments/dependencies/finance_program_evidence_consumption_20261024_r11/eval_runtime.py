import importlib.util
from pathlib import Path
path=Path(__file__).resolve().parent.parent/'finance_program_evidence_consumption_20261024_r10/eval_runtime.py'
spec=importlib.util.spec_from_file_location('r11_exact_r10_eval',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
ContextBackend,CPUReplay,raw_bytes=module.ContextBackend,module.CPUReplay,module.raw_bytes
TechnicalFailure,BudgetExhausted,TargetReached=module.TechnicalFailure,module.BudgetExhausted,module.TargetReached
prepare,preflight,require=module.prepare,module.preflight,module.require
# The inherited preflight uses this original parent's engine identity.
parent=module.parent
