"""Mode-aware bounded CPU loop; immutable native F66 parser/bridge are reused."""
from collections import Counter
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
STATE_IDS=('initial-09','continuation-01','recovery-02','recovery-06')
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
c=load('segment_original_collect20',ROOT.parent/'finance_program_teacher_state20_20261019_r1/collect20.py')


def expand(value,shared):
    if isinstance(value,dict):
        if set(value)=={'prompt_ref'}:return expand(shared[value['prompt_ref']],shared)
        return {k:expand(v,shared) for k,v in value.items()}
    if isinstance(value,list):return [expand(v,shared) for v in value]
    return deepcopy(value)


def seed(item,study):
    router=c.Router(item);ctrl=c.controller(router,study)
    from agent.orchestration.contracts import MetadataRecord
    record=MetadataRecord.from_mapping(item['initial'])
    original=ctrl.bridge.step;captured={};steps=[]
    def observe(call,record,state,counts,lineage,comparison_failures=None):
        result=original(call,record,state,counts,lineage,comparison_failures)
        captured.update(state=deepcopy(state),counts=deepcopy(counts),lineage=deepcopy(lineage),
                        comparison_entries=deepcopy(comparison_failures.entries if comparison_failures else {}))
        steps.append(deepcopy(result));return result
    ctrl.bridge.step=observe
    try:
        try:ctrl.run(item['initial'])
        except c.SnapshotReady:pass
    finally:ctrl.bridge.step=original
    if not router.target_exact or router.pos!=len(item['prefix']):raise ValueError('actual prefix/target mismatch')
    payload=json.loads(item['messages'][-1]['content'])
    state=expand(payload['current']['named_state'],payload.get('shared_values',{}))
    if not captured:
        expected={};ctrl.bridge.availability.initialize(record,expected)
        captured=dict(state=expected,counts=Counter(),lineage=dict(evidence=0,candidates=0,selection=0),comparison_entries={})
    if c.raw_json(captured['state'])!=c.raw_json(state):raise ValueError('captured state differs from exact native target')
    return dict(state=state,counts=dict(captured['counts']),lineage=captured['lineage'],
                comparison_entries=captured['comparison_entries'],prefix_steps=steps,
                feedback=expand(payload.get('feedback'),payload.get('shared_values',{})),
                target_exact=True,prefix_calls=router.pos,messages_sha256=c.sha(c.raw_json(item['messages'])))


def mode_boundary(mode,spec,status):
    if status!='success' or mode=='step':return True
    return mode=='segment' and ((spec.role=='metadata_acquisition' and spec.name!='SELECT_CONTEXT_FIELDS')
                                or 'standard_candidates' in spec.produces)


def check_mode(mode,plan):
    if mode not in ('whole','segment','step'):raise ValueError('unknown mode')
    if mode=='step' and len(plan.steps)!=1:raise ValueError('single Skill required; no automatic truncation')


MODE_TEXT={
    'whole':'Propose one complete remaining classification Program, within the remaining budget. Partial Programs are legal but do not by themselves complete classification.',
    'segment':'Propose only the next evidence-conditioned segment. End the segment after a metadata acquisition or production of standard_candidates so their actual outputs can inform the next decision. Do not commit a future tail beyond that boundary. Otherwise a supported category/grade/final suffix may run together.',
    'step':'Propose exactly ONE next Skill, using current verified observations. Do not output multiple steps. You will see its actual observation before deciding the next Skill.'}
COMMON=('Use only acquired source facts; taxonomy descriptions, guesses and provenance markers are not independent business evidence. '
        'Do not claim a specific product, record subject or ownership that current facts do not establish. '
        'If more information is needed, choose an available relevant acquisition, not repeated successful retrieval. '
        'Do not invent missing information merely to finish. Native contracts, typed outputs and terminal guards remain unchanged. '
        'Do not expose reasoning outside the required JSON. No reference answer or evaluation label is provided.')


def messages_for(ctrl,record,state,budget,feedback,mode):
    from program_loop.planner import planner_messages
    messages=planner_messages(record,state,ctrl.library,budget,feedback,skill_pool='F66')
    # One shared neutral base across arms; remove full-horizon objective conflicts.
    omit=('Plan the complete remaining classification task','The parser accepts partial Programs',
          'Consider the remaining complete task','F66 completion objective:',
          'Plan all dependencies through a successful terminal','You need not see a future output now')
    lines=messages[0]['content'].splitlines()
    messages[0]['content']='\n'.join(line for line in lines if not line.startswith(omit))
    messages[0]['content']+='\nPilot common evidence policy: '+COMMON+'\nPlanning mode: '+mode+'. '+MODE_TEXT[mode]
    payload=json.loads(messages[-1]['content']);payload['task']=MODE_TEXT[mode]
    payload['planning_mode']=mode
    if isinstance(payload.get('recovery_context'),dict):
        payload['recovery_context']['task_reminder']='Use planning_mode and the actual current evidence. Partial intent is not successful classification. No steps are supplied automatically.'
        payload['recovery_context']['tail_rule']='Unexecuted tail is historical intent, never required steps. Choose according to planning_mode and actual current evidence.'
    messages[-1]['content']=c.raw_json(payload).decode()
    return messages


def rollout(item,study,mode,generate,snapshot=None):
    if mode not in MODE_TEXT:raise ValueError('unknown mode')
    snapshot=deepcopy(snapshot if snapshot is not None else seed(item,study))
    local=deepcopy(study);local['budget']=dict(max_programs=6,max_program_steps=6,max_skill_steps=6,max_planner_attempts=6,max_identical_calls=1)
    ctrl=c.controller(generate,local)
    from agent.orchestration.contracts import MetadataRecord
    from program_loop.validation import parse_program
    from program_loop.failure_policy import ComparisonFailureLedger
    record=MetadataRecord.from_mapping(item['initial']);state=snapshot['state'];counts=Counter(snapshot['counts']);lineage=snapshot['lineage']
    failures=ComparisonFailureLedger();failures.entries=deepcopy(snapshot['comparison_entries'])
    feedback=snapshot['feedback']
    trace=dict(mode=mode,state_id=item['id'],source_id=item['source_id'],initial=deepcopy(item['initial']),
               seed_sha256=c.sha(c.raw_json(snapshot)),initial_state=deepcopy(state),planner_attempts=[],programs=[],
               new_steps=0,terminated=False,final_result=None,stop_reason='planner_budget_exhausted')
    for attempt_number in range(1,7):
        if trace['new_steps']>=6:trace['stop_reason']='skill_budget_exhausted';break
        if getattr(generate,'halted',False):trace['stop_reason']='transport_or_case_budget_halt';break
        remaining=dict(programs=7-attempt_number,planner_attempts=7-attempt_number,skill_steps=6-trace['new_steps'],program_steps=min(6,6-trace['new_steps']))
        messages=messages_for(ctrl,record,state,remaining,feedback,mode)
        attempt=dict(attempt=attempt_number,messages=deepcopy(messages),status='error',raw_response=None)
        trace['planner_attempts'].append(attempt)
        try:
            raw=generate(messages);attempt['raw_response']=raw;attempt['status']='rejected'
            plan=parse_program(raw,ctrl.bridge.catalog,state,remaining['program_steps']);check_mode(mode,plan)
        except Exception as exc:
            attempt['error_type']=type(exc).__name__
            attempt['error']=str(exc) if isinstance(exc,ValueError) else 'backend_error_details_retained_in_call_record'
            feedback=dict(status='planner_rejected',error=attempt['error'],remaining_budget=remaining)
            continue
        attempt['status']='accepted'
        obs=dict(program_index=len(trace['programs'])+1,attempt=attempt_number,plan=plan.to_dict(),steps=[],
                 status='completed',failure_index=None,terminated=False,invalidations=[])
        trace['programs'].append(obs)
        for index,call in enumerate(plan.steps):
            if trace['new_steps']>=6:obs.update(status='budget_exhausted',failure_index=index);break
            trace['new_steps']+=1
            step=ctrl.bridge.step(call,record,state,counts,lineage,failures);step['step_index']=index
            obs['steps'].append(step);obs['invalidations'].extend(step['invalidations'])
            if step['status']!='success':obs.update(status='failed',failure_index=index);break
            if ctrl.bridge.catalog[call.tool].terminal:
                trace.update(terminated=True,final_result=deepcopy(step['data']['final_result']),stop_reason='validated_terminal')
                obs.update(status='terminated',terminated=True);break
            if mode_boundary(mode,ctrl.bridge.catalog[call.tool],step['status']):
                obs['status']='completed';obs['yield_reason']='observed_evidence_boundary' if mode=='segment' else 'single_step_boundary';break
        obs['unexecuted_tail']=deepcopy(obs['plan']['program'][len(obs['steps']):])
        obs['remaining_budget']=dict(skill_steps=6-trace['new_steps'],planner_attempts=6-attempt_number,programs=6-attempt_number,program_steps=6)
        obs['lineage']=dict(lineage);feedback=deepcopy(obs)
        if trace['terminated']:break
    trace.update(state=deepcopy(state),final_lineage=dict(lineage),planner_calls=len(trace['planner_attempts']))
    return trace
