"""Response serialization instruction only; never change business content."""
import copy,json
FORMAT_RULE='Serialization requirement: Return exactly one valid JSON object, with double-quoted keys and JSON values, satisfying the existing output_schema. Do not return YAML, Markdown fences, or prose outside the JSON object. This requirement changes serialization only; do not add facts, infer missing information, or alter the existing evidence and business rules.'
TASK_RULE='Task clarification: Determine the field semantic category from actually acquired metadata; do not universally require actual row values or a category definition that literally repeats the field/table name. Definitions containing e.g., such as, 如 or 等 use non-exhaustive examples within their stated scope, not a closed vocabulary. Acquired descriptions are raw facts about intended field meaning, not observed transactions. A justified semantic comparison may map that meaning to a supplied definition, but model interpretation is not a new source fact. Keep the definition scope and exact-quote citation rules. If decision-critical subject or product ambiguity remains, report UNRESOLVED with that specific distinction; do not guess ownership, instrument type, values, or additional business context. Missing row values alone do not establish insufficiency. No answer or reference label is supplied.'
TASK_STAGES={'INFER_CATEGORY','INFER_CATEGORY_FROM_TOPK','ALIGN_FIELD_TO_STANDARD','ALIGN_DESCRIPTION_TO_STANDARD','COMPARE_STANDARD_CANDIDATES','FUSE_FIELD_EVIDENCE','FUSE_DATABASE_EVIDENCE','FUSE_TABLE_EVIDENCE','FUSE_STANDARD_EVIDENCE','CHECK_EVIDENCE_SUFFICIENCY','IDENTIFY_MISSING_EVIDENCE','SYNTHESIZE_METADATA_SEMANTICS'}
def format_messages(messages):
 out=copy.deepcopy(messages)
 if not (out and out[0]['role']=='system' and out[0]['content'].startswith('Execute only the selected Skill ')):return out
 stage=out[0]['content'].splitlines()[0].removeprefix('Execute only the selected Skill ').removesuffix('.')
 if stage in ('INFER_CATEGORY','INFER_CATEGORY_FROM_TOPK'):
  payload=json.loads(out[-1]['content'])
  lines=out[0]['content'].splitlines()
  lines=[x for x in lines if not x.startswith('Return JSON with `summary`, `data.category`') and not x.startswith('Return exactly one JSON object: {"option_id"')]
  out[0]['content']='\n'.join(lines)
  payload['output_schema']={'option_id':'one supplied short option ID or UNRESOLVED','summary':'short string; at most 20 characters','support':{'sufficient':'boolean','citations':[{'fact_id':'exact ID from evidence_boundary.facts','quote':'exact raw fact quote'}],'missing_evidence':'string; explain decision-critical gaps when insufficient'}}
  out[-1]['content']=json.dumps(payload,ensure_ascii=False,sort_keys=True,separators=(',',':'))
 elif stage in ('FUSE_FIELD_EVIDENCE','FUSE_DATABASE_EVIDENCE','FUSE_TABLE_EVIDENCE','FUSE_STANDARD_EVIDENCE'):
  payload=json.loads(out[-1]['content'])
  key={'FUSE_FIELD_EVIDENCE':'field_evidence','FUSE_DATABASE_EVIDENCE':'database_evidence','FUSE_TABLE_EVIDENCE':'table_evidence','FUSE_STANDARD_EVIDENCE':'standard_evidence'}[stage]
  if set(payload.get('output_schema',{}).get('data',{}))!={key}:raise ValueError('fusion schema must retain its exact original output key')
  payload['output_schema']={'summary':'non-empty concise natural-language observation','data':{key:'non-empty string; concise natural-language evidence state covering actual agreement, conflict and gaps; not an object or array'}}
  out[-1]['content']=json.dumps(payload,ensure_ascii=False,sort_keys=True,separators=(',',':'))
 elif stage=='SELECT_CONTEXT_FIELDS':
  payload=json.loads(out[-1]['content'])
  if set(payload.get('output_schema',{}).get('data',{}))!={'selected_context_fields'}:raise ValueError('selection schema must retain its exact original output key')
  payload['output_schema']={'summary':'non-empty concise explanation of selection and omissions; all explanation belongs here','data':{'selected_context_fields':'non-empty JSON array of distinct field-name strings drawn only from actually acquired table_fields; not objects or field descriptions; never invent names or values'}}
  out[-1]['content']=json.dumps(payload,ensure_ascii=False,sort_keys=True,separators=(',',':'))
 elif stage=='SYNTHESIZE_METADATA_SEMANTICS':
  payload=json.loads(out[-1]['content'])
  payload['output_schema']={'summary':'non-empty natural-language observation','data':{'metadata_semantics':'non-empty string; natural-language semantic state, not an object or array'}}
  out[-1]['content']=json.dumps(payload,ensure_ascii=False,sort_keys=True,separators=(',',':'))
 if stage in TASK_STAGES and TASK_RULE not in out[0]['content']:
  out[0]['content']=out[0]['content'].removesuffix('\n'+FORMAT_RULE)+'\n'+TASK_RULE
 if not out[0]['content'].endswith(FORMAT_RULE):out[0]['content']+='\n'+FORMAT_RULE
 return out
