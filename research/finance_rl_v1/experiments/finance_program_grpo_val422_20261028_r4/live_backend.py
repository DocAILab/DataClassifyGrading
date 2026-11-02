"""Historical greedy inference with the accepted GRPO actor and base-only Skills."""
import json
from pathlib import Path
from support import ROOT,BASE,digest
source=(BASE/'live_backend.py').read_text(encoding='utf-8')
if source.count('if index > 4200:')!=1:raise ValueError('historical claim budget contract changed')
original_check="('base_model_dir','max_sequence_tokens','scientific_updates_allowed')"
if source.count(original_check)!=1:raise ValueError('historical admission contract changed')
source=source.replace(original_check,"('base_model_dir','max_sequence_tokens')")
source=source.replace('if index > 4200:','if index > 17724:')
source=source.replace('100 cases times 42 maximum model calls exhausted','422 cases times 42 maximum model calls exhausted')
exec(compile(source,'<GRPO_Val422_historical_greedy>', 'exec'),globals())
HistoricalBackend=EvaluationBackend

class EvaluationBackend(HistoricalBackend):
    def __init__(self,admission):
        if admission.get('scientific_updates_allowed')!=0 or admission.get('maximum_optimizer_updates')!=0:raise ValueError('evaluation must forbid optimizer updates')
        super().__init__(admission)
        self.candidate=json.loads((ROOT/'candidate-request.json').read_bytes())
        self._candidate_loaded=False
    def _ensure_loaded(self):
        super()._ensure_loaded()
        if self._candidate_loaded:return
        import candidate_contract
        from model_backend import tensor_sha
        request=self.candidate
        root=(ROOT.parent/request['training_root']).resolve()
        if root.parent!=ROOT.parent.resolve() or root.name!='finance_program_grpo_full_migrated_20261027_r3':raise ValueError('training root identity mismatch')
        relative=Path(request['checkpoint_dir'])
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('unsafe checkpoint path')
        reference=digest(self.weight_shas)
        state=candidate_contract.load_checkpoint(root/relative,request['checkpoint_sha256'],reference,self.runtime['torch'])
        if state['policy_version']!=request['policy_version']:raise ValueError('accepted checkpoint version mismatch')
        candidate_contract.apply_checkpoint(self,state,tensor_sha,self.runtime['torch'],request['checkpoint_sha256'])
        self._candidate_loaded=True
