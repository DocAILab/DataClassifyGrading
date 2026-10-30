"""Wide-budget whole planner; native F66 contracts and observations unchanged."""
from collections import Counter
from copy import deepcopy
import importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location('wide_original_engine',ROOT.parent/'finance_program_teacher_segment_pilot_20261019_r1/engine.py')
old=importlib.util.module_from_spec(sp);sp.loader.exec_module(old)
def limits_remaining(attempt,new_steps):
    return dict(programs=7-attempt,planner_attempts=7-attempt,skill_steps=16-new_steps,program_steps=min(12,16-new_steps))
def rollout(item,study,generate,snapshot):
    snapshot=deepcopy(snapshot);local=deepcopy(study)
    local['budget']=dict(max_programs=6,max_program_steps=12,max_skill_steps=16,max_planner_attempts=6,max_identical_calls=1)
    ctrl=old.c.controller(generate,local)
    from agent.orchestration.contracts import MetadataRecord
    from program_loop.validation import parse_program
    from program_loop.failure_policy import ComparisonFailureLedger
    record=MetadataRecord.from_mapping(item['initial'])
    state=snapshot['state'];counts=Counter(snapshot['counts']);lineage=snapshot['lineage']
    failures=ComparisonFailureLedger();failures.entries=deepcopy(snapshot['comparison_entries'])
    feedback=snapshot['feedback']
    trace=dict(mode='whole',state_id=item['id'],source_id=item['source_id'],initial=deepcopy(item['initial']),
        seed_sha256=old.c.sha(old.c.raw_json(snapshot)),initial_state=deepcopy(state),planner_attempts=[],programs=[],
        new_steps=0,terminated=False,final_result=None,stop_reason='planner_budget_exhausted')
    for attempt_number in range(1,7):
        if trace['new_steps']>=16:trace['stop_reason']='skill_budget_exhausted';break
        if getattr(generate,'halted',False):trace['stop_reason']='transport_or_case_budget_halt';break
        remaining=limits_remaining(attempt_number,trace['new_steps'])
        messages=old.messages_for(ctrl,record,state,remaining,feedback,'whole')
        attempt=dict(attempt=attempt_number,messages=deepcopy(messages),status='error',raw_response=None)
        trace['planner_attempts'].append(attempt)
        try:
            raw=generate(messages);attempt['raw_response']=raw;attempt['status']='rejected'
            plan=parse_program(raw,ctrl.bridge.catalog,state,remaining['program_steps'])
        except Exception as exc:
            attempt['error_type']=type(exc).__name__
            attempt['error']=str(exc) if isinstance(exc,ValueError) else 'backend_error_details_retained_in_call_record'
            feedback=dict(status='planner_rejected',error=attempt['error'],remaining_budget=remaining);continue
        attempt['status']='accepted'
        obs=dict(program_index=len(trace['programs'])+1,attempt=attempt_number,plan=plan.to_dict(),steps=[],
            status='completed',failure_index=None,terminated=False,invalidations=[])
        trace['programs'].append(obs)
        for index,call in enumerate(plan.steps):
            if trace['new_steps']>=16:obs.update(status='budget_exhausted',failure_index=index);break
            trace['new_steps']+=1
            step=ctrl.bridge.step(call,record,state,counts,lineage,failures);step['step_index']=index
            obs['steps'].append(step);obs['invalidations'].extend(step['invalidations'])
            if step['status']!='success':obs.update(status='failed',failure_index=index);break
            if ctrl.bridge.catalog[call.tool].terminal:
                trace.update(terminated=True,final_result=deepcopy(step['data']['final_result']),stop_reason='validated_terminal')
                obs.update(status='terminated',terminated=True);break
        obs['unexecuted_tail']=deepcopy(obs['plan']['program'][len(obs['steps']):])
        obs['remaining_budget']=limits_remaining(attempt_number+1,trace['new_steps'])
        obs['lineage']=dict(lineage);feedback=deepcopy(obs)
        if trace['terminated']:break
    trace.update(state=deepcopy(state),final_lineage=dict(lineage),planner_calls=len(trace['planner_attempts']))
    return trace

