"""Same model backend, preserving the original bounded campaign accounting."""
import importlib.util
from common import ROOT,CAMPAIGN
spec=importlib.util.spec_from_file_location('same_evaluation_backend',CAMPAIGN/'backend.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
# Only the budget/disk anchor changes here: the original r4 campaign, never a reset.
module.ROOT=CAMPAIGN
ResourceFailure=module.ResourceFailure
class EvaluationBackend(module.EvaluationBackend):
    def __init__(self,admission):
        super().__init__(admission)
        self.identity['interface_root']=ROOT.name
        self.identity['campaign_root']=str(CAMPAIGN)
