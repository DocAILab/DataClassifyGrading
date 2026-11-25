"""V13: public standard descriptions for actual SELECT options, no target facts."""
import importlib.util,json
from pathlib import Path
path=Path(__file__).resolve().parent.parent/'finance_program_evidence_consumption_20261024_r11/interface.py'
spec=importlib.util.spec_from_file_location('v13_original_R11_interface',path);previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
GUIDANCE=('\nThe category_definition_index supplies the exact public definition for EACH current option ID. '
 'These are standard definitions, not target observations or preferred candidates. Rank exactly the original K '
 'options by their complete scope and the acquired source purpose; preserve all other contracts. '
 'Examples inside a definition are illustrative, not mandatory source field names. Neither this catalog nor '
 'a high rank establishes that the target satisfies a definition. No output, fact or grade is supplied.')
def rewrite(messages,*,session=None):
 actual=previous.rewrite(messages,session=session)
 if not actual[0]['content'].startswith('Execute only the selected Skill SELECT_CATEGORY_TOPK.'):return actual
 body=json.loads(actual[-1]['content'])
 catalog=json.loads((Path(__file__).resolve().parent/'standard-catalog.json').read_bytes())['entries']
 by={r['label']:r['description'] for r in catalog};assert len(by)==len(catalog)==12
 options=body['category_options'];assert len({r['option_id'] for r in options})==len(options)
 index=[]
 for option in options:
  assert set(option)=={'option_id','label'} and option['label'] in by,'public taxonomy/label identity mismatch'
  index.append(dict(option_id=option['option_id'],label=option['label'],description=by[option['label']]))
 body=dict(category_definition_index=index,**body)
 actual[-1]['content']=json.dumps(body,ensure_ascii=False,separators=(',',':'))
 actual[0]['content']+=GUIDANCE
 return actual
class RecoverySession(previous.RecoverySession):
 def rewrite(self,messages):return rewrite(messages,session=self)
class RecoveryAdapter(previous.RecoveryAdapter):
 def __init__(self,backend):super().__init__(backend);self.session=RecoverySession()
