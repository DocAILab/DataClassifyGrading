"""Keep R9/R5 native payload; add a per-rollout current-acquisition delta index."""
import importlib.util
import json
import re
from pathlib import Path
from evidence_delta import EvidenceDelta

path=Path(__file__).resolve().parent.parent/'finance_program_recovery_feedback_20261024_r5/interface.py'
spec=importlib.util.spec_from_file_location('consumption_exact_r9_interface',path)
previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
GUIDANCE=(
    '\nAcquired evidence review: acquired_evidence_delta is only a change index into the full current '
    'evidence_boundary.facts, not a conclusion. If a previous classification was insufficient and current '
    'raw observations have changed, read those exact facts before deciding whether the earlier distinction '
    'is still missing. Do not merely repeat a stale missing-evidence reason without comparing it with the '
    'actually acquired target and same-table source text. Candidate labels and earlier model interpretations '
    'are not new source facts. A returned neighbor field is evidence of that field, not automatically the '
    'target meaning; assess the relationship using actual source scope and purpose. '
    'Distinguish an explicitly stated entity/product/purpose from an unspecified generic attribute: an amount, '
    'time, name or transaction alone does not establish an unstated subject or product family. '
    'If the exact supplied definitions are supported, cite actual eligible fact IDs and exact quotes. '
    'If decisive uncertainty remains, keep UNRESOLVED and identify that specific remaining distinction, '
    'not unrelated absent metadata. No output is preselected, no historical gap is cleared by this index, '
    'and native support/citation/parser/persistent-gap/terminal guards remain unchanged.'
)

def rewrite(messages,*,session=None):
    actual=previous.rewrite(messages,session=session)
    match=re.match(r'Execute only the selected Skill ([A-Z_]+)\.',actual[0]['content'])
    if not match or match.group(1) not in ('INFER_CATEGORY','INFER_CATEGORY_FROM_TOPK'):
        return actual
    body=json.loads(actual[-1]['content'])
    tracker=session.evidence_delta if session is not None else EvidenceDelta()
    delta=tracker.project(body['evidence_boundary'])
    # Prepend pointers only, never duplicate the full fact values or manufacture support.
    body=dict(acquired_evidence_delta=delta,**body)
    actual[-1]['content']=json.dumps(body,ensure_ascii=False,separators=(',',':'))
    actual[0]['content']+=GUIDANCE
    return actual

class RecoverySession(previous.RecoverySession):
    def __init__(self):
        super().__init__()
        self.evidence_delta=EvidenceDelta()
    def rewrite(self,messages):return rewrite(messages,session=self)

class RecoveryAdapter(previous.RecoveryAdapter):
    def __init__(self,backend):
        super().__init__(backend)
        self.session=RecoverySession()
