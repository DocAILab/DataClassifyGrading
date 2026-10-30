import importlib.util
from common import PREVIOUS
spec=importlib.util.spec_from_file_location('same_accounted_backend',PREVIOUS/'backend.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
EvaluationBackend,ResourceFailure=module.EvaluationBackend,module.ResourceFailure
