"""Reuse trained N backend unchanged; all new calls use shared old ledger."""
import importlib.util,types
from common import ROOT,CAMPAIGN,put
spec=importlib.util.spec_from_file_location('n100_original_campaign_backend',CAMPAIGN/'backend.py');previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
ResourceFailure=previous.ResourceFailure
def claim_put(path,value):
 if path.parent.name=='call-claims':value=dict(value,version=3,evaluation='student_n100',execution_root=ROOT.name)
 return put(path,value)
class EvaluationBackend(previous.EvaluationBackend):
 def __init__(self,admission):
  super().__init__(admission);self.identity.update(version=3,global_version=8,prior_same_round_actual_calls=168,evaluation='N-only100',frozen_interface='finance_program_evidence_consumption_20261024_r11')
method=previous.EvaluationBackend.generate
generator=types.FunctionType(method.__code__,dict(method.__globals__,put=claim_put),method.__name__,method.__defaults__,method.__closure__)
def generate(self,messages,arm,stage,destination):
 if arm!='N':raise ValueError('N100 has no base/F evaluation arm')
 return generator(self,messages,arm,stage,destination)
EvaluationBackend.generate=generate
