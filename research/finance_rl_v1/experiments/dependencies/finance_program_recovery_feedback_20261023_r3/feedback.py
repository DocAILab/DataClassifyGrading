"""Project only already public observations and contracts; never resolve private data."""
from copy import deepcopy
import json

def resolve(value,shared,seen=()):
    if isinstance(value,dict) and set(value)=={'prompt_ref'}:
        key=value['prompt_ref']
        if key in seen or key not in shared:return None
        return resolve(shared[key],shared,seen+(key,))
    return value

def project(body,prior_gaps=()):
    shared=body.get('shared_values',{})
    current=resolve(body.get('current',{}),shared) or {}
    state=resolve(current.get('named_state',{}),shared) or {}
    initial=current.get('input',{})
    rules=body.get('shared_input_rules',{})
    recovery=resolve(body.get('recovery_context',{}),shared) or {}
    feedback=resolve(body.get('feedback',{}),shared) or {}
    observations=resolve(recovery.get('availability_observations',[]),shared) or []
    contracts={};options=[];preparations=[]
    def presence(key,seen=()):
        if key in seen:return False
        rule=rules.get(key,{})
        ref=rule.get('canonical_reference')
        value=state.get(key) if not rule else None
        if isinstance(ref,str) and ref.startswith('$state.'):
            value=state
            for part in ref[7:].split('.'):
                value=resolve(value,shared)
                value=value.get(part) if isinstance(value,dict) else None
        elif ref=='$metadata.field_name':value=initial.get('field_name')
        elif ref=='$domain':value=initial.get('domain')
        value=resolve(value,shared)
        present=value is not None and value!='' and value!=[] and value!={}
        native_binding=(ref=='$record' and rule.get('source')=='initial_binding'
            and rule.get('explicit_value_policy')=='initial_binding_contract')
        required=deepcopy(rule.get('requires_all',[]))
        missing=[k for k in required if not presence(k,seen+(key,))]
        contracts[key]=dict(canonical_reference=ref,present_in_public_state=present,
            binding_status='native_initial_binding' if native_binding else ('public_value_present' if present else 'not_present'),
            binding_available=present or native_binding,
            requires_all=required,missing_requires_all=missing,
            possible_producers=deepcopy(rule.get('possible_producers',[])),presence_is_not_readiness=True)
        return (present or native_binding) and not missing
    library=body.get('skill_library',[])
    preparation_names=set()
    pending={k for s in library if s.get('role')=='metadata_acquisition' for k in s.get('required_inputs',[])}|{'context_field_semantics','context_semantics','standard_alignment_evidence'}
    for _ in range(len(library)):
        added=[s for s in library if s['name'] not in preparation_names and not s.get('terminal') and set(s.get('produces',[]))&pending]
        if not added:break
        preparation_names.update(s['name'] for s in added)
        pending.update(k for s in added for k in s.get('required_inputs',[]))
    for skill in library:
        if skill.get('role')!='metadata_acquisition' and skill['name'] not in preparation_names:continue
        required=skill.get('required_inputs',[])
        missing=[k for k in required if not presence(k)]
        row=dict(name=skill['name'],required_inputs=deepcopy(required),missing_required_inputs=missing,produces=deepcopy(skill.get('produces',[])),
            observations=[{k:o.get(k) for k in ('scope','type','status')} for o in observations if o.get('skill')==skill['name']],
            output_guaranteed=False)
        if skill.get('role')=='metadata_acquisition':options.append(row)
        if skill['name'] in preparation_names:preparations.append(row)
    gaps=[dict(deepcopy(g),historical_report=True,resolution='not_reassessed_by_this_projection') for g in prior_gaps];failures=[]
    for step in resolve(feedback.get('steps',[]),shared) or []:
        step=resolve(step,shared) or {}
        if step.get('status') not in ('success',None):
            failures.append(dict(skill=(step.get('call') or {}).get('tool'),status=step.get('status'),summary=step.get('summary')))
        for audit in step.get('evidence_audits',[]):
            support=audit.get('support') or {}
            if support.get('sufficient') is False and support.get('missing_evidence'):
                gaps.append(dict(skill=audit.get('skill'),missing_evidence=support['missing_evidence'],gap_index=audit.get('gap_index')))
    # A current report replaces the same historical report; neither is a new adjudication.
    unique={}
    for gap in gaps:
        key=(gap.get('skill'),json.dumps(gap.get('missing_evidence'),ensure_ascii=False,sort_keys=True))
        unique[key]=gap
    choices=resolve(state.get('missing_metadata_types',[]),shared)
    return dict(schema='public-recovery-feedback-v1',unresolved_gaps=list(unique.values()),latest_failures=failures,
        planner_rejection=feedback.get('error'),acquisition_options=options,preparation_options=preparations,input_contracts=contracts,
        metadata_acquisition_choices=deepcopy(choices),
        remaining_budget=deepcopy(body.get('remaining_budget',{})),evidence_added=False,automatic_actions=False,
        meaning='Options are copied from public contracts, not recommendations or guaranteed evidence. Presence does not establish readiness; canonical unavailable and unknown differ.')
