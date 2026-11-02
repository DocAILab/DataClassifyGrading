"""New prospectively bounded round; all earlier429 calls remain explicit history."""
import importlib.util
from common import ROOT,ORIGINAL,put
spec=importlib.util.spec_from_file_location('unchanged_fresh162_model_backend',ORIGINAL/'backend.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
def accounted_put(path,value):
    if path.parent==ROOT/'campaign/call-claims':
        index=int(path.stem);value=dict(value,round=2,round_request=index,prior_round_claimed=429,global_request=429+index)
    put(path,value)
module.put=accounted_put
ResourceFailure=module.ResourceFailure
class EvaluationBackend(module.EvaluationBackend):
    def __init__(self,admission):
        super().__init__(admission)
        self.identity.update(interface_root=ROOT.name,campaign_root=str(ROOT),prior_actual_calls=429,round=2,
            acquisition_contract='declared-named-source-group-v1',physical_database_names_invented=0)
