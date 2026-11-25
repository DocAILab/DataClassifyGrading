"""Acquired-only request views and structural citation checks, not a truth oracle."""
from copy import deepcopy
import hashlib
import json
import re

IDENTIFIERS={'field_name','table_name','database_name','domain'}
BUSINESS_LEAVES={'field_description','table_description','table_descriptions','database_description','field_type','value'}
CLASSIFIERS={'INFER_CATEGORY','INFER_CATEGORY_FROM_TOPK'}
POLICY=('Evidence boundary v1: evidence_boundary.facts contains only actually observed source values. '
        'Identifiers/abbreviations, unrelated tables in the same database, model interpretations, category definitions and repeated explanations are not independent target-field business evidence. '
        'Use hypotheses only as uncertain suggestions; never call an inferred field prefix an observed table name. '
        'A missing metadata field is not automatically required. If a decision-critical distinction is unresolved, '
        'choose a relevant available acquisition or report insufficient evidence, not invented facts or repeated successful calls. '
        'Known unavailable differs from unqueried. Reuse obtained facts. No target answer is provided.')

def raw_json(value):
    return json.dumps(value,ensure_ascii=False,separators=(',',':'),allow_nan=False)

def digest(value):return hashlib.sha256(raw_json(value).encode('utf-8')).hexdigest()

def strict(raw):
    def pairs(items):
        out={}
        for k,v in items:
            if k in out:raise ValueError('support_schema: duplicate key')
            out[k]=v
        return out
    def nonfinite(_):raise ValueError('support_schema: nonfinite')
    value=json.loads(raw,object_pairs_hook=pairs,parse_constant=nonfinite)
    raw_json(value).encode('utf-8',errors='strict')
    return value

def evidence_view(record,state):
    from pool_runtime import assemble_pool_evidence
    from pool_metadata import FIELDS
    bundle=assemble_pool_evidence(record,state)
    facts={}
    for key,value in bundle['metadata'].items():
        facts['metadata.'+key]=dict(value=deepcopy(value),provenance=bundle['provenance'].get(key,'initial_record'),
            decisive_eligible=key in BUSINESS_LEAVES,role='raw_fact',source_scope='target_record')
    for key,value in bundle['observations'].items():
        provenance=bundle['provenance'].get(key)
        prefix='observations.'+key
        # Composite wrappers include identifiers, coverage and counts. Their mere
        # acquisition does not make arbitrary substrings decisive business facts.
        facts[prefix]=dict(value=deepcopy(value),provenance=provenance,decisive_eligible=False,role='raw_fact',source_scope='auxiliary_composite')
        def leaves(obj,path,business=False,target_scope=key!='database_context'):
            if isinstance(obj,dict):
                if key=='database_context' and 'table_name' in obj:
                    target_scope=bool(bundle['metadata'].get('table_name')) and obj['table_name']==bundle['metadata']['table_name']
                for child,v in obj.items():leaves(v,path+'.'+child,child in BUSINESS_LEAVES,target_scope)
            elif isinstance(obj,list):
                for index,v in enumerate(obj):leaves(v,path+'.'+str(index),business,target_scope)
            elif obj is not None:
                facts[path]=dict(value=deepcopy(obj),provenance=provenance,decisive_eligible=bool(business and target_scope),role='raw_fact',
                    source_scope='target_or_same_table_context' if target_scope else 'same_database_auxiliary')
        leaves(value,prefix)
    hypotheses={k:deepcopy(v) for k,v in bundle['roles'].items()
        if v['role']!='raw_fact'}
    ledger=deepcopy(state.get('availability_ledger',[]))
    obtained={k for k in bundle['metadata']}
    missing=[]
    for key in sorted(set(FIELDS)-obtained):
        observation=next((z for z in ledger if z.get('scope')=='record' and z.get('type')==key),None)
        status=observation.get('status','unknown') if observation else 'unknown'
        missing.append(dict(metadata_type=key,status=status,required_for_this_decision='not_automatically_required'))
    return dict(schema='acquired-evidence-boundary-v1',facts=facts,hypotheses=hypotheses,
        missing_metadata=missing,availability_ledger=ledger,
        authority='Presence/valid quote is not proof that a category inference follows. Read actual source support.',
        scope='No private rows, reference labels or unqueried metadata are exposed.')

def complete_optional(call,record,state,catalog):
    from pool_runtime import _binding
    from agent.orchestration.contracts import ToolCall
    from agent.orchestration.validator import resolve_reference
    spec=catalog[call.tool];arguments=deepcopy(call.arguments);audit=[]
    for key in spec.optional_inputs:
        if key in arguments:continue
        reference=_binding(key,record,state)
        if reference is None:continue
        found,value=resolve_reference(reference,record,state)
        if not found:continue
        arguments[key]=reference
        audit.append(dict(argument=key,reference=reference,value_sha256=digest(value),kind='observed_optional_input'))
    return ToolCall(call.tool,arguments),audit

def classification_response(raw,view):
    value=strict(raw)
    if not isinstance(value,dict) or set(value)!={'option_id','summary','support'}:
        raise ValueError('support_schema: exact classification keys required')
    if any(not isinstance(value[k],str) or not value[k].strip() for k in ('option_id','summary')):
        raise ValueError('support_schema: nonempty native fields required')
    support=value['support']
    if not isinstance(support,dict) or set(support)!={'sufficient','citations','missing_evidence'}:
        raise ValueError('support_schema: exact support keys required')
    if type(support['sufficient']) is not bool or not isinstance(support['missing_evidence'],str):
        raise ValueError('support_schema: typed sufficiency required')
    citations=support['citations']
    if not isinstance(citations,list) or len(citations)>8:raise ValueError('support_schema: bounded citations required')
    if not support['sufficient']:
        if citations:raise ValueError('support_schema: insufficient outcome requires empty citations')
        if not support['missing_evidence'].strip():raise ValueError('support_schema: missing-evidence reason required')
        raise ValueError('insufficient: '+support['missing_evidence'][:500])
    if not citations:raise ValueError('citation: at least one decisive acquired fact required')
    seen=set()
    for citation in citations:
        if not isinstance(citation,dict) or set(citation)!={'fact_id','quote'}:
            raise ValueError('support_schema: exact citation keys required')
        fid,quote=citation['fact_id'],citation['quote']
        if not isinstance(fid,str) or fid not in view['facts'] or fid in seen:raise ValueError('citation: unknown or duplicate fact ID')
        seen.add(fid);fact=view['facts'][fid]
        if fact['decisive_eligible'] is not True:raise ValueError('identifier_or_auxiliary: not decisive target evidence')
        text=fact['value'] if isinstance(fact['value'],str) else raw_json(fact['value'])
        if not isinstance(quote,str) or not quote.strip() or len(quote)>512 or quote not in text:
            raise ValueError('quote: must literally occur in the acquired fact')
    return raw_json({k:value[k] for k in ('option_id','summary')}),deepcopy(support)

def skill_name(messages):
    match=re.match(r'Execute only the selected Skill ([A-Z_]+)\.',messages[0]['content'])
    return match.group(1) if match else None

class Guard:
    def __init__(self,generate):
        self.generate=generate;self.record=None;self.state=None;self.audits=[]
    @property
    def halted(self):return getattr(self.generate,'halted',False)
    @property
    def last_generation_metadata(self):return getattr(self.generate,'last_generation_metadata',{})
    def __call__(self,messages):
        actual=deepcopy(messages);payload=json.loads(actual[-1]['content']);name=skill_name(actual)
        if name is None:
            from agent.orchestration.contracts import MetadataRecord
            def expand(value):
                if isinstance(value,dict):
                    if set(value)=={'prompt_ref'}:return expand(payload['shared_values'][value['prompt_ref']])
                    return {k:expand(v) for k,v in value.items()}
                if isinstance(value,list):return [expand(v) for v in value]
                return value
            record=MetadataRecord.from_mapping(payload['current']['input'])
            view=evidence_view(record,expand(payload['current']['named_state']))
        else:
            if self.record is None or self.state is None:raise ValueError('guard context missing')
            view=evidence_view(self.record,self.state)
        actual[0]['content']+='\n'+POLICY
        payload['evidence_boundary']=view
        if name in CLASSIFIERS:
            actual[0]['content']+=('\nVersioned output extension: return exactly option_id, summary, support. '
                'support={sufficient:boolean,citations:[{fact_id:string,quote:string}],missing_evidence:string}. '
                'For sufficient=true cite at least one decisive_eligible raw fact with an exact quote. '
                'Do not pretend a literal quote proves your inference. If a subject/product distinction is unsupported, '
                'return sufficient=false, empty citations, missing_evidence explaining it and option_id="UNRESOLVED". '
                'This is a legitimate insufficient outcome, not permission to guess a category.')
            payload['output_extension']='support is required; only evidence_boundary.facts may be cited'
        actual[-1]['content']=raw_json(payload)
        audit=dict(skill=name or 'Planner',view_sha256=digest(view),
            actual_messages_sha256=digest(actual),classification=name in CLASSIFIERS,status='requested')
        self.audits.append(audit)
        try:raw=self.generate(actual)
        except Exception as exc:
            audit.update(status='generation_error',error_kind='raw_format' if str(exc)=='restricted_format_rejected' else 'backend_error')
            raise
        audit['provider_response_sha256']=hashlib.sha256(raw.encode('utf-8')).hexdigest()
        if name not in CLASSIFIERS:
            audit['status']='native_response';return raw
        try:
            native,support=classification_response(raw,view)
        except (ValueError,UnicodeError,TypeError) as exc:
            text=str(exc)
            kind='insufficient' if text.startswith('insufficient:') else 'citation_rejected' if text.startswith(('citation:','identifier:','identifier_or_auxiliary:','quote:')) else 'support_schema'
            audit.update(status='rejected',error_kind=kind,reason=text[:600])
            try:audit['support']=strict(raw).get('support')
            except (ValueError,UnicodeError,AttributeError):pass
            raise ValueError('evidence_boundary_'+text) from None
        audit.update(status='validated_citations',support=support,native_response_sha256=hashlib.sha256(native.encode('utf-8')).hexdigest())
        return native
