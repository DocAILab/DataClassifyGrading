"""Emphasize only the current request's public candidate-ID definitions."""
import copy,importlib.util,json,re
from pathlib import Path
path=Path(__file__).resolve().parent.parent/'finance_program_recovery_feedback_20261024_r4/interface.py'
spec=importlib.util.spec_from_file_location('r4_verified_interface',path)
previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
def rewrite(messages,*,session=None):
    actual=previous.rewrite(messages,session=session)
    stage=re.match(r'Execute only the selected Skill ([A-Z_]+)\.',actual[0]['content'])
    if not stage or stage.group(1) not in ('SELECT_CATEGORY_TOPK','INFER_CATEGORY','INFER_CATEGORY_FROM_TOPK'):return actual
    body=json.loads(actual[-1]['content'])
    candidates=body.get('category_options') if stage.group(1)=='SELECT_CATEGORY_TOPK' else body.get('candidate_details')
    if not isinstance(candidates,list):return actual
    body=dict(candidate_scope_index=copy.deepcopy(candidates),candidate_scope_evidence_added=False,**body)
    actual[-1]['content']=json.dumps(body,ensure_ascii=False,separators=(',',':'))
    actual[0]['content']+= ('\nCandidate scope check: IDs are opaque identifiers, not ranks or a semantic family. '
        'Read the exact full path paired with EACH supplied ID in candidate_scope_index. '
        'Do not treat a contiguous range of IDs as one family or silently exclude supplied enterprise, '
        'system-operation or user definitions because the database domain is finance. '
        'Compare only the definitions/labels actually supplied here with the acquired field AND table descriptions. '
        'For Top-K, the first catalog entries are not a default recommendation: rank by observed semantic purpose, '
        'respect exactly K, and do not fill all slots with asset subclasses merely because an asset type is unknown. '
        'For final inference, explicitly consider all supplied candidate families, not just financial assets versus accounting. '
        'The index repeats public definitions; it adds no target fact and no preferred answer. '
        'Do not force a residual/other category when its own scope lacks support. '
        'Missing row values/types are not decision-critical by themselves; genuine missing subject/product or internal-versus-customer '
        'context remains insufficient. Preserve UNRESOLVED when no supplied option has actual decisive support. '
        'Only native eligible raw facts and exact citations support a sufficient outcome; unchanged parser and evidence guards apply.')
    return actual
class RecoverySession(previous.RecoverySession):
    def rewrite(self,messages):return rewrite(messages,session=self)
class RecoveryAdapter(previous.RecoveryAdapter):
    def __init__(self,backend):
        super().__init__(backend);self.session=RecoverySession()
