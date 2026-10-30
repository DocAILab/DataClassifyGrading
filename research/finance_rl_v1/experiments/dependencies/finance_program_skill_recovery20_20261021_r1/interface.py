"""Observed-only binding feedback. No output repair, plan veto or invented data."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('typed_contracts20',ROOT.parent/'finance_program_skill_interface_smoke_20261021_r2/next_contracts.py')
typed=importlib.util.module_from_spec(spec);spec.loader.exec_module(typed)
RULE=('Binding interface: binding_status describes ONLY facts already in this student state, not future query results. '
      'An absent field_type/value is unbound: do not add its interpretation merely to complete a six-step template. '
      'GET_FIELD_METADATA may supply omitted source attributes, but absent attributes may remain unavailable. '
      'After its result, reuse actual existing metadata; do not repeat an obtained/unavailable query. '
      'An acquired context with returned_count=0 or fields=[] supplies NO new business support; missing_locators explain why. '
      'Never infer a missing database, join across unidentified databases, treat a hypothesis as raw evidence, '
      'or repeat classification after insufficient without a genuinely new relevant source fact. '
      'If evidence remains inadequate, preserve insufficient; no forced final answer. '
      'The provided remaining_budget is authoritative: there is NO six-Skill requirement or ceiling. '
      'Plan as many necessary steps as the explicit budget permits; avoid unnecessary steps. '
      'Return only the existing Program JSON format; no new stop/reason fields.')

def rewrite(messages):
    result=typed.rewrite(messages,'unified')
    if re.match(r'Execute only the selected Skill ([A-Z_]+)\.',result[0]['content']):return result
    payload=json.loads(result[-1]['content']);view=payload.get('evidence_boundary',{})
    facts=view.get('facts',{});keys={k.removeprefix('metadata.') for k,f in facts.items()
        if k.startswith('metadata.') and f.get('value') not in (None,'')}
    keys.update(k for k,v in payload.get('current',{}).get('input',{}).items() if v not in (None,''))
    empty=[]
    for fid in ('observations.table_fields','observations.table_context','observations.database_context'):
        value=facts.get(fid,{}).get('value')
        if isinstance(value,dict) and (value.get('returned_count')==0 or value.get('fields')==[] or value.get('tables')==[]):
            empty.append(dict(fact_id=fid,missing_locators=deepcopy(value.get('missing_locators',[])),
                              business_evidence_acquired=False))
    payload['binding_status']=dict(observed_metadata_keys=sorted(keys),
        unbound_interpretations=[name for name,key in [('INTERPRET_FIELD_TYPE','field_type'),('INTERPRET_FIELD_VALUE','value')]
            if key not in keys],empty_acquisitions=empty,authority='observed state only, not future availability')
    result[0]['content']+='\n'+RULE
    result[-1]['content']=json.dumps(payload,ensure_ascii=False,separators=(',',':'),allow_nan=False)
    return result
