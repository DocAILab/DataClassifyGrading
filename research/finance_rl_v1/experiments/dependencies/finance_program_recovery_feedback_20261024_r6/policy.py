import importlib.util
from common import CAMPAIGN
spec=importlib.util.spec_from_file_location('same_fixed_policy',CAMPAIGN/'policy.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
gate_slots,gate_pass,pending_slots=module.gate_slots,module.gate_pass,module.pending_slots
