"""Single ablation: selector sees a compact faithful index of current raw facts."""
import copy,importlib.util,json
from pathlib import Path
path=Path(__file__).resolve().parent.parent/'finance_program_evidence_consumption_20261024_r10/interface.py'
spec=importlib.util.spec_from_file_location('r11_exact_r10_interface',path);previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
GUIDANCE=('\nSource-first candidate review: selector_source_index repeats ALL current eligible source values with exact fact IDs, '
 'grouped only by their native scope, without judging relevance. Read these acquired target descriptions and table context '
 'when ranking the supplied catalog labels. semantic_query and domain_semantics are earlier model interpretations, not an '
 'exclusive domain restriction; a finance database can contain any of the supplied enterprise, system or user data families. '
 'Determine the observed purpose/environment from the full current source, not just the isolated target attribute or first '
 'catalog entries. Use same-table descriptions to understand the table environment, without transferring a neighbor value '
 'to the target. An actual source description can establish a named context without sample row values; a generic attribute '
 'alone does not establish an unnamed subject/product. Rank exactly K existing IDs yourself. No preferred ID, inferred '
 'purpose, new fact, sufficiency verdict or output is supplied by this index. Retain all original uncertainty and guards.')
def rewrite(messages,*,session=None):
 actual=previous.rewrite(messages,session=session)
 if not actual[0]['content'].startswith('Execute only the selected Skill SELECT_CATEGORY_TOPK.'):return actual
 body=json.loads(actual[-1]['content']);facts=body['evidence_boundary']['facts']
 index=dict(schema='all-current-eligible-source-index-v1',target_record=[],same_table_context=[],sufficiency_adjudicated=False,facts_added=False)
 for fid,fact in facts.items():
  if fact.get('role')!='raw_fact' or fact.get('decisive_eligible') is not True:continue
  scope=fact.get('source_scope')
  if scope=='target_record':index['target_record'].append([fid,copy.deepcopy(fact['value'])])
  elif scope=='target_or_same_table_context':index['same_table_context'].append([fid,copy.deepcopy(fact['value'])])
  else:raise ValueError('unexpected eligible source scope; do not silently discard')
 body=dict(selector_source_index=index,**body)
 actual[-1]['content']=json.dumps(body,ensure_ascii=False,separators=(',',':'))
 actual[0]['content']+=GUIDANCE
 return actual
class RecoverySession(previous.RecoverySession):
 def rewrite(self,messages):return rewrite(messages,session=self)
class RecoveryAdapter(previous.RecoveryAdapter):
 def __init__(self,backend):super().__init__(backend);self.session=RecoverySession()
