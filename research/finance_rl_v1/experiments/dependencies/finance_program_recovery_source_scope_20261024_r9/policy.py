import importlib.util
from common import ORIGINAL
spec=importlib.util.spec_from_file_location('unchanged_fixed20_gate',ORIGINAL/'policy.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
gate_slots,gate_pass,pending_slots=module.gate_slots,module.gate_pass,module.pending_slots
