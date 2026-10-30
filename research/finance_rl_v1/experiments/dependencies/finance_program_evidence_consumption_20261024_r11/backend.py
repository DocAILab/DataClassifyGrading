"""Exact inherited generator, same campaign; record this additive version honestly."""
import importlib.util,types
from common import PREVIOUS,put
spec=importlib.util.spec_from_file_location('r11_original_model_backend',PREVIOUS/'backend.py');previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
ResourceFailure=previous.ResourceFailure
def claim_put(path,value):
 if path.parent.name=='call-claims':value=dict(value,version=2)
 return put(path,value)
class EvaluationBackend(previous.EvaluationBackend):
 def __init__(self,admission):
  super().__init__(admission);self.identity.update(version=2,prior_same_round_actual_calls=83,global_version=7)
method=previous.EvaluationBackend.generate
namespace=dict(method.__globals__,put=claim_put)
EvaluationBackend.generate=types.FunctionType(method.__code__,namespace,method.__name__,method.__defaults__,method.__closure__)
