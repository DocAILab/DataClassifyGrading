"""New bounded N-only inference window using the unchanged final162 adapter."""
import json
import sys
import time
from support import ROOT, TRAIN, ADAPTER_SHA, native, put

native()
sys.path.append(str(TRAIN))
from model_backend import ModelBackend
from runtime_safety import guard_resources


class ResourceFailure(BaseException):
    pass


class EvaluationBackend(ModelBackend):
    def __init__(self, admission):
        original = json.loads((TRAIN/'admission.json').read_bytes())
        super().__init__(original)
        assert self.expected['adapter_model.safetensors'] == ADAPTER_SHA
        assert admission['evaluation_only'] and admission['gpu_deadline_unix'] > time.time()
        for key in ('base_model_dir','max_sequence_tokens','scientific_updates_allowed'):
            assert admission[key] == original[key]
        self.admission = admission
        self.identity.update(evaluation_window_separate_from_training=True,
                             training_admission=original,evaluation_admission=admission,
                             evaluation_root=ROOT.name,only_student_arm=True,
                             termination_policy='termination-first-v1')

    def generate(self, messages, arm, stage, destination):
        from paired_eval import BudgetExhausted
        if arm != 'N':
            raise ValueError('only the new Student SFT algorithm is authorized')
        self.last_generation_metadata = {}
        self.last_call_evidence = {}
        if time.time() >= self.admission['gpu_deadline_unix'] - 20:
            raise BudgetExhausted('termination-first evaluation absolute deadline exhausted')
        try:
            guard_resources(ROOT,self.runtime['torch'])
        except RuntimeError as exc:
            raise ResourceFailure(str(exc)) from exc
        claims = ROOT/'campaign/call-claims'
        claims.mkdir(parents=True,exist_ok=True)
        index = len(list(claims.glob('*.json'))) + 1
        if index > 4200:
            raise BudgetExhausted('100 cases times 42 maximum model calls exhausted')
        put(claims/f'{index:06d}.json',dict(index=index,arm=arm,stage=stage,
                                         destination=str(destination),time_unix=time.time()))
        try:
            return super().generate(messages,arm,stage,destination)
        except Exception as exc:
            if 'out of memory' in str(exc).lower() or type(exc).__name__ in ('OutOfMemoryError','HostMemoryWait'):
                raise ResourceFailure(str(exc)) from exc
            raise
