"""Common dynamic controller, scoped metadata paths and native semantic execution."""
import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

from agent.orchestration.agent_loop import (AgentLoopController, LoopConfig, LoopState, PolicyDecision,
    ActionError, parse_policy_action, build_final_evidence_state)
from agent.orchestration.category_catalog import CategoryCatalog
from agent.orchestration.contracts import ToolCall
from agent.orchestration.model_skill_executor import (ModelSkillExecutor, SkillExecutionError,
    parse_skill_observation, _generation_metadata)
from agent.orchestration.runtime import Observation
from agent.orchestration.skill_loader import SkillLoader
from pool_catalog import ROOT, pool_catalog, purpose, SEARCH_NAMES
from pool_metadata import PoolMetadata, FIELDS, present
from pool_bindings import bind_call, BINDING_RULE_VERSION
from comparison_output import parse_comparison_output, ComparisonOutputError, MAX_REASON_LENGTH
from evidence_roles import acquired_source, evidence_roles
from grounded_semantics import TARGETS as GROUNDED_TARGETS, valid_shape as grounded_shape, make_messages as grounded_messages, validate_observation as validate_grounded

POLICY_SYSTEM = '''You are a database-field classification orchestration policy.
Choose exactly one currently callable Skill, using the global capability index to plan a minimum sufficient evidence program.
Only domain and field_name are visible initially. Other raw metadata requires explicit acquisition; pending metadata is not evidence.
Choose the Skill; the controller binds required state arguments. Omit controller-bound arguments, or supply their exact advertised references or strictly identical values. Explicit values are checked before normalization.
ACQUIRE_MISSING_METADATA requires an explicit missing_metadata_type chosen from actual missing_metadata_types. Optional arguments may be supplied only when actually bound. Never use provenance source names as argument data.
Never invent raw metadata, standard entries, labels or grading levels. Intermediate semantic strings are evidence, not acquired raw values.
Stop acquiring when evidence is sufficient, select a supported category, confirm its frozen grade and finalize within the remaining budget.
Return exactly one JSON object: {"tool":"SKILL_NAME","arguments":{}}. No Program, Markdown, plain stop, or extra JSON.'''

SEMANTIC_KEYS = (
    'domain_semantics','field_abbreviation_semantics','field_semantics','disambiguated_field_semantics',
    'field_description_semantics','field_type_semantics','value_semantics','table_abbreviation_semantics',
    'table_semantics','table_description_semantics','database_abbreviation_semantics','database_semantics',
    'database_description_semantics','context_field_semantics','metadata_semantics',
    'field_evidence','context_evidence','table_evidence','database_evidence','metadata_evidence',
)
ALIGNMENT_KEYS=('field_standard_alignment','context_standard_alignment','table_standard_alignment',
    'database_standard_alignment','value_standard_alignment','description_standard_alignment','standard_comparison')

# Every validated, declared-type native output participates in the evidence
# boundary, including refinement, conflict resolution and support assessments.
SEMANTIC_KEYS=tuple(dict.fromkeys((*SEMANTIC_KEYS,*(key for spec in pool_catalog('F66').values()
    if spec.role in ('semantic_interpretation','evidence_reasoning') for key in spec.produces
    if key not in ('category','grading_level','aligned_standard_entry','final_result')))))
ALIGNMENT_KEYS=tuple(dict.fromkeys((*ALIGNMENT_KEYS,*(key for spec in pool_catalog('F66').values()
    if spec.role=='standard_alignment' for key in spec.produces
    if key not in ('standard_candidates','aligned_standard_entry')))))


def valid_evidence_value(key,value):
    if key in GROUNDED_TARGETS.values():
        return grounded_shape(value)
    if key=='evidence_sufficiency':
        return (isinstance(value,Mapping) and set(value)=={'sufficient','rationale'}
            and type(value['sufficient']) is bool
            and isinstance(value['rationale'],str) and bool(value['rationale'].strip()))
    return isinstance(value,str) and bool(value.strip())


def validated_evidence_values(state,keys):
    catalog=pool_catalog('F66');provenance=state.get('_provenance',{})
    return {key:state[key] for key in keys
        if key in state and provenance.get(key) in catalog
        and key in catalog[provenance[key]].produces
        and valid_evidence_value(key,state[key])}


def acquired_raw(record,state):
    provenance=state.get('_provenance',{})
    return {'field_name':record.field_name,**{k:state[k] for k in FIELDS
        if k!='field_name' and acquired_source(k,provenance.get(k)) and present(state.get(k))}}


def assemble_pool_evidence(record,state):
    """Single allowlisted boundary for publication, model search and finalization."""
    provenance=state.get('_provenance',{})
    semantics=validated_evidence_values(state,SEMANTIC_KEYS)
    alignments=validated_evidence_values(state,ALIGNMENT_KEYS)
    observations={k:state[k] for k in ('table_fields','selected_context_fields','context_fields',
        'context_field_descriptions','context_field_types','context_field_values','table_context','database_context')
        if acquired_source(k,provenance.get(k)) and present(state.get(k))}
    metadata=acquired_raw(record,state) if record is not None else dict(state.get('current_metadata',{}))
    names=set(metadata)|set(semantics)|set(alignments)|set(observations)
    prior=state.get('available_evidence',{})
    return {'domain':record.domain if record is not None else prior.get('domain'),
        'field_name':record.field_name if record is not None else metadata.get('field_name'),
        'metadata':metadata,'semantics':semantics,'alignments':alignments,'observations':observations,
        'provenance':{k:provenance[k] for k in sorted(names) if k in provenance},
        'roles':evidence_roles(metadata,semantics,alignments,observations,provenance)}


def pool_final_evidence(record,state):
    final=build_final_evidence_state(state)
    if final is not None:
        evidence=assemble_pool_evidence(record,state)
        final['semantic_evidence']={**evidence['semantics'],**evidence['alignments']}
        final['available_evidence']=evidence
    return final


def publish_state(record,state,data,skill):
    """Publish only typed observed data and explicit evidence bundles, with provenance."""
    spec=pool_catalog('F66').get(skill)
    if spec is None or not set(data)<=set(spec.produces):
        raise SkillExecutionError('publication contains undeclared outputs or unknown Skill')
    for key,value in data.items():
        if key in (*SEMANTIC_KEYS,*ALIGNMENT_KEYS) and not valid_evidence_value(key,value):
            raise SkillExecutionError(f'{key} violates its declared evidence type')
    state.update(data)
    provenance=state.setdefault('_provenance',{})
    provenance.update({k:skill for k in data})
    if isinstance(data.get('metadata_availability'),Mapping):
        state['current_metadata']=acquired_raw(record,state)
        state['available_metadata']={'domain':record.domain,**state['current_metadata']}
        missing=data['metadata_availability'].get('missing',[])
        state['missing_metadata_types']=[k for k in missing if k in FIELDS]
        if state['missing_metadata_types']:
            state['missing_metadata_type']=state['missing_metadata_types'][0]
            provenance['missing_metadata_type']=skill
        else:
            state.pop('missing_metadata_type',None)
            provenance.pop('missing_metadata_type',None)
    raw={}
    for key in ('field_metadata','acquired_metadata'):
        value=data.get(key)
        if isinstance(value,Mapping):
            raw.update({k:v for k,v in value.items() if k in FIELDS and present(v)})
    if skill.startswith('GET_'):
        raw.update({k:v for k,v in data.items() if k in FIELDS and present(v)})
    for key,value in raw.items():
        state[key]=value; provenance[key]=skill
    if raw:
        # Initial field_name belongs to the record, not necessarily to state.
        # Publish the same acquired-only boundary consumed by the executor.
        state['current_metadata']=acquired_raw(record,state)
        state['available_metadata']={'domain':record.domain,**state['current_metadata']}
        state['missing_metadata_types']=[k for k in FIELDS if k not in state['current_metadata']]
        if state['missing_metadata_types']:
            state['missing_metadata_type']=state['missing_metadata_types'][0]
            provenance['missing_metadata_type']=skill
        else:
            state.pop('missing_metadata_type',None)
            provenance.pop('missing_metadata_type',None)
    aliases={'context_field_semantics':'context_semantics','field_semantics':'candidate_field_semantics',
             'category':'aligned_category','grading_level':'grading_level_candidate'}
    for original,alias in aliases.items():
        if original in state and present(state[original]):
            state[alias]=state[original]; provenance[alias]=provenance.get(original,skill)
    if 'category' in state:
        for alias in ('candidate_category','category_candidate'):
            state[alias]=state['category']; provenance[alias]=provenance.get('category',skill)
    if isinstance(state.get('standard_candidates'),list) and state['standard_candidates']:
        for alias in ('candidate_standard_categories','candidate_categories','previous_standard_results'):
            state[alias]=state['standard_candidates']; provenance[alias]=provenance.get('standard_candidates',skill)
    semantics=validated_evidence_values(state,SEMANTIC_KEYS)
    alignments=validated_evidence_values(state,ALIGNMENT_KEYS)
    groups={
        'available_field_evidence':{k:v for k,v in semantics.items() if k.startswith(('field_','disambiguated_','value_'))},
        'context_field_evidence':{k:v for k,v in semantics.items() if k.startswith('context_')},
        'available_table_evidence':{k:v for k,v in semantics.items() if k.startswith('table_')},
        'available_database_evidence':{k:v for k,v in semantics.items() if k.startswith('database_')},
        'available_metadata_evidence':semantics,'standard_alignment_evidence':alignments,
        'aligned_standard_candidates':alignments,'available_semantic_context':{k:v for k,v in semantics.items() if k not in ('field_semantics','field_abbreviation_semantics')},
    }
    if semantics:
        for name in ('semantic_evidence','current_semantic_state','semantic_query','current_evidence',
                     'current_evidence_state','all_available_evidence','evidence','evidence_set'):
            groups[name]=semantics
    # Multiple model interpretations do not establish a business-data conflict.
    if len(alignments)>=2:
        groups['conflicting_standard_alignments']=alignments
    for name,value in groups.items():
        if value:
            state[name]=value; provenance[name]='validated-evidence-bundle'
    # Raw retrieved metadata is also usable search evidence, without pretending it is semantics.
    if isinstance(state.get('field_metadata'),Mapping) or semantics:
        state['available_evidence']=assemble_pool_evidence(record,state)
        provenance['available_evidence']='validated-evidence-bundle'
        state['all_available_evidence']=state['available_evidence']
        provenance['all_available_evidence']='validated-evidence-bundle'
        if 'semantic_query' not in state:
            state['semantic_query']=state['available_evidence']
    evidence=pool_final_evidence(record,state)
    if evidence is not None:
        state['final_evidence_state']=evidence
        provenance['final_evidence_state']='frozen-classification-bundle'


def _binding(name,record,state):
    if name=='current_record': return '$record'
    if name=='domain': return '$domain'
    if name in ('field_name','target_field'): return '$metadata.field_name'
    if name=='table_fields':
        value=state.get(name)
        if not isinstance(value,Mapping) or value.get('coverage')!='canonical-val-same-table-only' or value.get('missing_locators')!=[]:
            return None
        fields=value.get('fields')
        if (not isinstance(fields,list) or not fields or len(fields)>30
            or type(value.get('returned_count')) is not int or value['returned_count']!=len(fields)
            or type(value.get('available_count')) is not int or value['available_count']<len(fields)
            or not all(present(value.get(k)) for k in ('database_name','table_name'))
            or not all(isinstance(f,Mapping) and isinstance(f.get('field_name'),str) and present(f['field_name']) for f in fields)):
            return None
    if name=='selected_context_fields':
        value=state.get(name)
        if not isinstance(value,list) or not value or not all(isinstance(v,str) and present(v) for v in value) or len(set(value))!=len(value):
            return None
    if name=='context_fields' and (not isinstance(state.get(name),list) or not state[name] or not all(isinstance(f,Mapping) and present(f.get('field_name')) for f in state[name])):
        return None
    if name=='candidate_standard_categories' and present(state.get('standard_candidates')):
        return '$state.standard_candidates'
    if name in state and present(state[name]): return '$state.'+name
    if name=='all_available_evidence' and present(state.get('available_evidence')): return '$state.available_evidence'
    return None


def derive_pool_callable(record,state,catalog):
    named=state.named_state if isinstance(state,LoopState) else state
    result={}
    for name,spec in catalog.items():
        if all(_binding(k,record,named) is not None for k in spec.required_inputs):
            # Placeholders, empty structures and unprovenanced raw state cannot enable raw Skills.
            raw_inputs=set(spec.required_inputs)&(set(FIELDS)-{'field_name'})
            raw_inputs|=set(spec.required_inputs)&{'table_fields','context_fields','selected_context_fields'}
            if any(k not in named.get('_provenance',{}) for k in raw_inputs):
                continue
            result[name]=spec
    return result


class PoolPolicy:
    def __init__(self,generate,catalog,*,missing_awareness=False,metadata_prefetch=False):
        self.generate=generate; self.catalog=catalog
        self.missing_awareness=missing_awareness
        self.metadata_prefetch=metadata_prefetch

    def select_action(self,record,state,callable_skills,*,feedback=None,forced_finalize=False):
        if (self.metadata_prefetch and not forced_finalize and state.turns==0
                and not any(c.get('tool')=='GET_FIELD_METADATA' for c in state.call_history)
                and 'GET_FIELD_METADATA' in callable_skills):
            call=ToolCall('GET_FIELD_METADATA',{'current_record':'$record'})
            return PolicyDecision(call,json.dumps(call.to_dict(),ensure_ascii=False),
                metadata={'fixed_metadata_prefetch':True})
        index=[{'name':s.name,'purpose':purpose(s.name),'required_inputs':list(s.required_inputs),
                'optional_inputs':list(s.optional_inputs),'produces':list(s.produces),'terminal':s.terminal} for s in self.catalog.values()]
        payload={'task':'Predict the target leaf category and its frozen standard grading level.',
            'current':state.to_model_dict(),'capability_index':index,'callable_skills':list(callable_skills),
            'argument_bindings':{name:{k:_binding(k,record,state.named_state) for k in (*spec.required_inputs,*spec.optional_inputs)
                if k!='missing_metadata_type' and _binding(k,record,state.named_state) is not None} for name,spec in callable_skills.items()},
            'pending_metadata_retrieval':'Optional metadata is unavailable until explicitly retrieved from canonical Val. Missing types/values/descriptions remain unavailable.',
            'forced_finalize':forced_finalize,'output_schema':{'tool':'SKILL_NAME','arguments':{}}}
        if feedback: payload['validation_feedback']=feedback
        if self.missing_awareness:
            from pool_availability import RULE
            payload['availability_rule']=RULE
        messages=[{'role':'system','content':POLICY_SYSTEM},{'role':'user','content':json.dumps(payload,ensure_ascii=False,indent=2)}]
        try:
            raw=self.generate(messages)
            call=parse_policy_action(raw)
        except Exception as exc:
            if isinstance(exc,ActionError):
                exc.metadata={'messages':messages,'raw_response':locals().get('raw'),'generation':_generation_metadata(self.generate)}
                raise
            error=ActionError(f'policy generator failed: {exc}')
            error.metadata={'messages':messages,'raw_response':None,'generation':_generation_metadata(self.generate)}
            raise error from exc
        return PolicyDecision(call,raw,tuple(messages),_generation_metadata(self.generate))


class PoolExecutor(ModelSkillExecutor):
    def __init__(self,generate,loader,metadata_lookup,category_catalog,*,top1_selection_mode='model',include_comparison_explanation=True):
        super().__init__(generate,loader,category_catalog=category_catalog,top_k=5,guard_two_stage_evidence=True,
            evidence_assembly=assemble_pool_evidence,top1_selection_mode=top1_selection_mode,
            include_comparison_explanation=include_comparison_explanation)
        self.metadata_lookup=metadata_lookup

    def comparison_inputs(self,record,state,resolved):
        """The actual comparison model input, shared with recovery fingerprinting."""
        if resolved['standard_candidates']!=state.get('standard_candidates'):
            raise SkillExecutionError('alignment must consume actual prior candidates')
        ids=[item['category'] for item in resolved['standard_candidates']]
        # The model chooses a short ID; only the executor binds the frozen object.
        return {**resolved,'standard_candidate_details':self.category_catalog.option_details(ids),
                'available_evidence':assemble_pool_evidence(record,state)}

    def _execute_comparison(self,record,state,resolved,contract,*,correction_context=None):
        inputs=self.comparison_inputs(record,state,resolved)
        system=(f'Execute only the selected Skill {contract.name}.\n'
            f'DSL: {contract.dsl}\nContract:\n{contract.body}\n'
            'Use only the resolved inputs. Return exactly one compact JSON decision with '
            'decision, option_id, reason; no Markdown or extra fields.')
        payload={'skill':contract.name,'resolved_inputs':inputs,'output_schema':{
            'decision':'match or no_match',
            'option_id':'one exact short option_id from standard_candidate_details for match; null for no_match',
            'reason':f'non-empty brief string, at most {MAX_REASON_LENGTH} characters'}}
        if correction_context is not None:
            payload['correction_context']=correction_context
        messages=[{'role':'system','content':system},
                  {'role':'user','content':json.dumps(payload,ensure_ascii=False,indent=2)}]
        raw=None
        try:
            raw=self.generate(messages)
        except Exception as exc:
            generation=_generation_metadata(self.generate)
            raise SkillExecutionError(str(exc),metadata={**getattr(exc,'metadata',{}),
                **generation,'messages':messages,'raw_response':None,'generation':generation,
                'failure_kind':'backend_error'}) from exc
        generation=_generation_metadata(self.generate)
        metadata={**generation,'messages':messages,'raw_response':raw,'generation':generation}
        if generation.get('truncation_at_budget') is True:
            raise SkillExecutionError('comparison generation reported truncation at output budget',
                metadata={**metadata,'failure_kind':'output_truncated'})
        try:
            decision=parse_comparison_output(raw,[item['option_id'] for item in inputs['standard_candidate_details']])
            if decision['decision']=='no_match':
                raise SkillExecutionError(decision['reason'],metadata={'failure_kind':'no_match'})
            category=self.category_catalog.resolve_option_ids([decision['option_id']],1)[0]
            observation=Observation(contract.name,'success',decision['reason'],{
                'standard_comparison':decision['reason'],
                'aligned_standard_entry':self.category_catalog.frozen_entry(category)})
            observation=self._validate_semantic(observation,self.loader.catalog[contract.name],state)
        except (ComparisonOutputError,SkillExecutionError,ValueError) as exc:
            failure_kind=getattr(exc,'metadata',{}).get('failure_kind','invalid_output')
            raise SkillExecutionError(str(exc),metadata={**getattr(exc,'metadata',{}),**metadata,
                'failure_kind':failure_kind}) from exc
        except Exception as exc:
            raise SkillExecutionError(str(exc),metadata={**metadata,'failure_kind':'unknown_error'}) from exc
        return replace(observation,metadata=metadata)

    def _scoped_inputs(self,record,resolved):
        for key in ('table_name','database_name','field_name','target_field'):
            if key not in resolved: continue
            actual=record.field_name if key=='target_field' else self.metadata_lookup.get(record).get(key)
            if resolved[key]!=actual:
                raise SkillExecutionError(f'{key} differs from actual privately scoped record')

    def _metadata(self,call,record,resolved,state):
        lookup=self.metadata_lookup; own=lookup.get(record); name=call.tool
        self._scoped_inputs(record,resolved)
        if name=='CHECK_METADATA_AVAILABILITY':
            # Report currently visible raw inputs, not secretly loaded source data.
            visible={'domain':record.domain,**acquired_raw(record,state)}
            data={'metadata_availability':{'available':list(visible),'missing':[k for k in FIELDS if k not in visible]}}
        elif name=='GET_FIELD_METADATA': data={'field_metadata':own}
        elif name in ('GET_FIELD_DESCRIPTION','GET_TABLE_DESCRIPTION','GET_DATABASE_DESCRIPTION'):
            key=self.loader.catalog[name].produces[0]
            if key not in own: return Observation(name,'unavailable',f'Actual canonical metadata has no {key}',{},metadata={'missing':[key],'source':'canonical-val'})
            data={key:own[key]}
        elif name=='GET_TABLE_FIELDS': data={'table_fields':lookup.table_fields(record)}
        elif name=='GET_COMPLETE_TABLE_CONTEXT': data={'table_context':lookup.table_context(record)}
        elif name=='GET_COMPLETE_DATABASE_CONTEXT': data={'database_context':lookup.database_context(record)}
        elif name=='EXPAND_SELECTED_CONTEXT': data={'context_fields':lookup.expand(record,resolved['selected_context_fields'])}
        elif name in ('GET_CONTEXT_FIELD_DESCRIPTIONS','GET_CONTEXT_FIELD_TYPES','GET_CONTEXT_FIELD_VALUES'):
            fields=resolved['context_fields']
            names=[f['field_name'] if isinstance(f,Mapping) else f for f in fields] if isinstance(fields,list) else fields
            expanded=lookup.expand(record,names)
            rawkey={'GET_CONTEXT_FIELD_DESCRIPTIONS':'field_description','GET_CONTEXT_FIELD_TYPES':'field_type','GET_CONTEXT_FIELD_VALUES':'value'}[name]
            missing=[f['field_name'] for f in expanded if rawkey not in f]
            available=[{'field_name':f['field_name'],rawkey:f[rawkey]} for f in expanded if rawkey in f]
            if not available: return Observation(name,'unavailable',f'No actual context {rawkey} is available',{},metadata={'missing_fields':missing,'source':'canonical-val'})
            data={self.loader.catalog[name].produces[0]:{'fields':available,'missing_fields':missing}}
        elif name=='ACQUIRE_MISSING_METADATA':
            key=resolved['missing_metadata_type']
            if not isinstance(key,str) or key not in FIELDS: raise SkillExecutionError('requested metadata type is outside allowlist')
            if key not in own: return Observation(name,'unavailable',f'Canonical source has no requested {key}',{},metadata={'missing':[key],'source':'canonical-val'})
            data={'acquired_metadata':{key:own[key]}}
        else: raise SkillExecutionError(f'unsupported metadata acquisition {name}')
        return Observation(name,'success',f'{purpose(name)} Actual source data loaded with bounded coverage.',data,metadata={'source':'canonical-val'})

    def _validate_semantic(self,observation,spec,state):
        data=dict(observation.data)
        for key,value in observation.data.items():
            if key=='aligned_standard_entry':
                if not isinstance(value,Mapping): raise SkillExecutionError('aligned entry must be an actual frozen object')
                category=value.get('category')
                prior=state.get('standard_candidates',[])
                if not isinstance(category,str) or not any(isinstance(c,Mapping) and c.get('category')==category for c in prior):
                    raise SkillExecutionError('aligned entry must belong to actual retrieved candidates')
                frozen=self.category_catalog.frozen_entry(category)
                if spec.name=='COMPARE_STANDARD_CANDIDATES':
                    visible=self.category_catalog.details([category])[0]
                    if not ((set(value)==set(visible) and value==visible) or
                            (set(value)==set(frozen) and value==frozen)):
                        raise SkillExecutionError('aligned entry differs from authoritative candidate views')
                    data[key]=frozen
                elif value!=frozen: raise SkillExecutionError('aligned entry differs from frozen Registry')
                if state.get('category') and state['category']!=category: raise SkillExecutionError('comparison cannot overwrite selected category alignment')
            elif not valid_evidence_value(key,value):
                raise SkillExecutionError(f'{key} violates its declared evidence type')
        return replace(observation,data=data)

    def _validate_consumed_raw(self,record,resolved,state):
        """Bind every supplied raw argument to acquired canonical values before generation."""
        own=self.metadata_lookup.get(record)
        provenance=state.get('_provenance',{})
        for key,value in resolved.items():
            if key=='current_record':
                if value!={'domain':record.domain,'metadata':dict(record.metadata)}:
                    raise SkillExecutionError('current_record differs from actual initial record')
            elif key in ('domain','field_name','target_field'):
                actual=record.domain if key=='domain' else record.field_name
                if value!=actual: raise SkillExecutionError(f'{key} differs from actual initial record')
            elif key in FIELDS:
                if key not in provenance or key not in state or value!=state[key] or value!=own.get(key):
                    raise SkillExecutionError(f'{key} must equal actually acquired canonical raw metadata')
            elif key=='current_metadata':
                if value!=state.get(key) or value!=acquired_raw(record,state):
                    raise SkillExecutionError('current_metadata must equal actually acquired raw state')
            elif key=='available_metadata':
                actual={'domain':record.domain,**acquired_raw(record,state)}
                if not isinstance(value,Mapping) or value!=state.get(key) or value!=actual:
                    raise SkillExecutionError('available_metadata must equal actually acquired public raw metadata')
            elif key=='table_fields':
                if key not in provenance or value!=state.get(key) or value!=self.metadata_lookup.table_fields(record) or _binding(key,record,state) is None:
                    raise SkillExecutionError('table_fields must equal acquired nonempty same-table candidates')
            elif key=='context_fields':
                if key not in provenance or value!=state.get(key) or _binding(key,record,state) is None:
                    raise SkillExecutionError('context_fields must equal acquired expanded context objects')
                actual=self.metadata_lookup.expand(record,[f['field_name'] for f in value])
                if value!=actual: raise SkillExecutionError('context_fields differ from actual bounded neighbors')
            elif key=='selected_context_fields':
                if key not in provenance or value!=state.get(key) or _binding(key,record,state) is None:
                    raise SkillExecutionError('selected_context_fields must equal actual acquired selection')
                self.metadata_lookup.expand(record,value)

    def execute(self,call,record,state,contract,*,correction_context=None):
        if self.loader.load(call.tool)!=contract: raise SkillExecutionError('contract mismatch')
        spec=self.loader.catalog[call.tool]
        allowed=set(spec.required_inputs)|set(spec.optional_inputs)
        if set(call.arguments)-allowed or any(k not in call.arguments for k in spec.required_inputs):
            raise SkillExecutionError('arguments must satisfy the selected native contract')
        resolved=self._resolved_arguments(call,record,state)
        if any(not present(resolved[k]) for k in spec.required_inputs): raise SkillExecutionError('required inputs must be nonempty actual values')
        self._validate_consumed_raw(record,resolved,state)
        if spec.role=='metadata_acquisition' and call.tool!='SELECT_CONTEXT_FIELDS':
            return self._validate_deterministic(self._metadata(call,record,resolved,state),spec)
        if call.tool in SEARCH_NAMES:
            query=resolved[spec.required_inputs[0]]
            internal=ToolCall('SELECT_CATEGORY_TOPK',{'semantic_query':query})
            # The internal engine role is compatible; contract/header remains requested native Skill.
            observation=self._execute_two_stage(internal,record,{'semantic_query':query},state,contract)
            return replace(observation,skill=call.tool,metadata={**observation.metadata,'standard_backend':'model-catalog-top5-v1','requested_skill':call.tool})
        if call.tool=='INFER_CATEGORY':
            internal=ToolCall('INFER_CATEGORY_FROM_TOPK',call.arguments)
            observation=self._execute_two_stage(internal,record,resolved,state,contract)
            return replace(observation,skill=call.tool,metadata={**observation.metadata,'requested_skill':call.tool})
        if call.tool=='INFER_GRADING_LEVEL':
            category=resolved['aligned_category']; entry=resolved['aligned_standard_entry']
            if category!=state.get('category') or entry!=state.get('aligned_standard_entry') or entry!=self.category_catalog.frozen_entry(category):
                raise SkillExecutionError('grade must consume actual selected canonical standard entry')
            return Observation(call.tool,'success',f'Frozen standard {category} assigns {entry["grading_level"]}',{'grading_level':entry['grading_level']},metadata={'controller_bound_grade':True})
        if call.tool in ('SELECT_CATEGORY_TOPK','INFER_CATEGORY_FROM_TOPK','FINALIZE_CLASSIFICATION'):
            return super().execute(call,record,state,contract)
        if call.tool=='SELECT_CONTEXT_FIELDS':
            self._scoped_inputs(record,resolved)
            actual=self.metadata_lookup.table_fields(record)
            if resolved['table_fields']!=actual: raise SkillExecutionError('context selection must consume actual same-table candidates')
        if call.tool=='COMPARE_STANDARD_CANDIDATES':
            return self._execute_comparison(record,state,resolved,contract,correction_context=correction_context)
        # Alignment evidence receives actual descriptions for already selected candidates.
        if spec.role=='standard_alignment' and 'standard_candidates' in resolved:
            if resolved['standard_candidates']!=state.get('standard_candidates'): raise SkillExecutionError('alignment must consume actual prior candidates')
            ids=[item['category'] for item in resolved['standard_candidates']]
            resolved={**resolved,'standard_candidate_details':self.category_catalog.details(ids)}
        grounding_context = None
        if call.tool in GROUNDED_TARGETS:
            actual_raw = acquired_raw(record,state)
            grounding_context = {k: actual_raw[k] for k in ('field_name','field_description','table_name','table_description') if k in actual_raw}
            messages = grounded_messages(contract,resolved,grounding_context)
        else:
            messages=self._messages(contract,resolved)
        try:
            raw=self.generate(messages)
            observation=parse_skill_observation(raw,spec)
            if grounding_context is not None:
                validate_grounded(observation, grounding_context)
            if call.tool=='SELECT_CONTEXT_FIELDS':
                self.metadata_lookup.expand(record,observation.data['selected_context_fields'])
            else:
                observation=self._validate_semantic(observation,spec,state)
        except Exception as exc:
            if isinstance(exc,SkillExecutionError):
                exc.metadata={'messages':messages,'raw_response':locals().get('raw'),**_generation_metadata(self.generate)}
                raise
            raise SkillExecutionError(str(exc),metadata={'messages':messages,'raw_response':locals().get('raw'),**_generation_metadata(self.generate)}) from exc
        return replace(observation,metadata={'messages':messages,'raw_response':raw,**_generation_metadata(self.generate)})


class PoolController(AgentLoopController):
    def _normalize_policy_decision(self,decision,record,state,spec):
        call,audit=bind_call(decision.call,spec,record,state,_binding)
        if self.policy.missing_awareness:
            from pool_availability import blocked
            denied=blocked(call,state)
            if denied: raise ActionError(f'availability repeat target blocked: {denied}')
        return replace(decision,call=call,metadata={**decision.metadata,'binding':audit})

    def _rejected_policy_metadata(self,decision,error):
        return {'binding':{'version':BINDING_RULE_VERSION,'status':'rejected',
            'raw_call':decision.call.to_dict() if decision is not None else None,
            'normalized_call':None,'changes':[]}}

    def _available_evidence(self,record,state):
        return assemble_pool_evidence(record,state)

    def _final_evidence_state(self,record,state):
        return pool_final_evidence(record,state)

    def _normalize_observation(self,call,observation,state):
        if self.policy.missing_awareness:
            from pool_availability import observe
            observe(call,observation,state)
        if observation.status=='success' and call.tool in ('INFER_CATEGORY','INFER_CATEGORY_FROM_TOPK'):
            category=observation.data.get('category')
            if category in self.registry_entries:
                frozen={'category':category,**self.registry_entries[category]}
                if observation.data.get('aligned_standard_entry')!=frozen:
                    return Observation(call.tool,'error','category alignment differs from frozen Registry',{})
                # Both native and compatibility category engines bind the same canonical entry.
                normalized=super()._normalize_observation(call,observation,state)
                if normalized.status=='success':
                    return replace(normalized,data={**normalized.data,'aligned_standard_entry':frozen})
                return normalized
        return super()._normalize_observation(call,observation,state)


def build_pool_controller(generate,registry,metadata_path,pool,config,*,missing_awareness=False,
                          metadata_prefetch=False,top1_selection_mode='model'):
    """Build P14 controls; fixed prefetch needs at least 2 budgeted turns."""
    if type(missing_awareness) is not bool: raise ValueError('missing_awareness must be boolean')
    if type(metadata_prefetch) is not bool: raise ValueError('metadata_prefetch must be boolean')
    if type(top1_selection_mode) is not str or top1_selection_mode not in ('model','first'):
        raise ValueError('top1_selection_mode must be model or first')
    if pool!='P14' and (metadata_prefetch or top1_selection_mode!='model'):
        raise ValueError('nondefault controls require P14')
    catalog=pool_catalog(pool)
    if isinstance(registry,(str,Path)):
        registry=json.loads(Path(registry).read_text(encoding='utf-8'))
    if isinstance(registry,Mapping) and 'categories' in registry:
        entries={item['category_id']:{k:v for k,v in item.items() if k not in ('category','category_id')} for item in registry['categories']}
    elif isinstance(registry,Mapping): entries={k:dict(v) for k,v in registry.items()}
    else: raise ValueError('registry must be a frozen category mapping or Registry file')
    loader=SkillLoader(ROOT/'runtime/skills/full_pool',catalog,strict_catalog=False)
    loader.validate_all()  # Installation is a separate explicit pre-freeze action.
    metadata=PoolMetadata.from_canonical(metadata_path)
    category_catalog=CategoryCatalog(entries)
    executor=PoolExecutor(generate,loader,metadata,category_catalog,top1_selection_mode=top1_selection_mode)
    if config is None:
        config=LoopConfig(recover_ready_finalization=True)
    elif isinstance(config,Mapping):
        config=LoopConfig(**config)
    if metadata_prefetch and config.max_turns<2:
        raise ValueError('metadata_prefetch requires at least 2 turns')
    def derive(record,state,catalog):
        from pool_availability import initialize,filter_callable
        named=state.named_state if isinstance(state,LoopState) else state
        initialize(record,named)
        return filter_callable(named,derive_pool_callable(record,state,catalog))
    return PoolController(policy=PoolPolicy(generate,catalog,missing_awareness=missing_awareness,
        metadata_prefetch=metadata_prefetch),executor=executor,catalog=catalog,
        skill_loader=loader,config=config,registry_entries=entries,
        approved_levels={e['grading_level'] for e in entries.values()},
        callable_derivation=derive if missing_awareness else derive_pool_callable,state_publication=publish_state)
