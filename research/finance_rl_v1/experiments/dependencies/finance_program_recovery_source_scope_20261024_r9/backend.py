"""Evaluate already completed fresh162 weights under a new explicit evaluation window."""
import json,sys,time
from common import ROOT,PARENT,CAMPAIGN,put
sys.path.append(str(PARENT))
from model_backend import ModelBackend
from runtime_safety import guard_resources
class ResourceFailure(BaseException):pass
class EvaluationBackend(ModelBackend):
    def __init__(self,admission):
        original=json.loads((PARENT/'admission.json').read_bytes())
        # Audit the historical training completion against its true old admission.
        super().__init__(original)
        assert self.expected['adapter_model.safetensors']=='6d78e440ab0c9749427e002d54fd94cca06dea77c31b63ae3628b48fdbec4e64'
        for key in ('base_model_dir','max_sequence_tokens','scientific_updates_allowed'):
            assert admission[key]==original[key],key
        assert admission['evaluation_only'] is True and admission['gpu_deadline_unix']>time.time()
        self.admission=admission
        self.identity.update(evaluation_window_separate_from_training=True,training_admission=original,
            evaluation_admission=admission,interface_root=ROOT.name,campaign_root=str(CAMPAIGN),
            prior_round_actual_calls=429,round=2,version=2,acquisition_contract='declared-named-source-group-v1')
    def generate(self,messages,arm,stage,destination):
        from paired_eval import BudgetExhausted
        self.last_generation_metadata={};self.last_call_evidence={}
        try:
            if time.time()>=self.admission['gpu_deadline_unix']-20:raise BudgetExhausted('new campaign deadline exhausted')
            try:guard_resources(ROOT,self.runtime['torch'])
            except RuntimeError as exc:raise ResourceFailure(str(exc)) from exc
            claims=CAMPAIGN/'campaign/call-claims';claims.mkdir(exist_ok=True,parents=True)
            index=len(list(claims.glob('*.json')))+1
            if index>5040:raise BudgetExhausted('finite campaign5040 request budget exhausted')
            put(claims/f'{index:06d}.json',dict(time_unix=time.time(),arm=arm,stage=stage,destination=str(destination),
                round=2,round_request=index,prior_round_claimed=429,global_request=429+index,version=2))
            return super().generate(messages,arm,stage,destination)
        except Exception as exc:
            if 'out of memory' in str(exc).lower() or type(exc).__name__ in ('OutOfMemoryError','HostMemoryWait'):raise ResourceFailure(str(exc)) from exc
            raise
