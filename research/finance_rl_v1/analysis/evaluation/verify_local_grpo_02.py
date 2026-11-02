"""Independent exact replay and actual model-input/route checks after download."""
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from support import N100, ADAPTER_SHA, load, native, put, sha
from generation_audit_policy import generation_status,outcome_status
from actual_input_audit import validate_actual_input
from trace_audit_02 import same_trace


def main():
    import torch
    assert not torch.cuda.is_initialized()
    ex=ROOT/'reviews/final-01/extract'
    receipt=json.loads((ROOT/'reviews/final-01/receipt.json').read_bytes())
    assert sha(ROOT/'reviews/final-01/original.tar.gz')==receipt['archive_sha256']
    for row in receipt['members']:
        p=ex/row['path']
        assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256']
    assert json.loads((ex/'launch/exit.json').read_bytes())['actual_returncode']==0
    assert json.loads((ex/'supervision/worker-exit.json').read_bytes())['actual_returncode']==0
    completion=json.loads((ex/'runs/completion.json').read_bytes())
    assert len(completion['rows'])==422 and completion['N_only'] and completion['reused_results']==0
    request=json.loads((ROOT/'candidate-request.json').read_bytes())
    state_path=ROOT.parent/request['training_root']/'reviews/final-01/extract'/request['checkpoint_dir']/'state.pt'
    assert sha(state_path)==request['checkpoint_sha256']
    state=torch.load(state_path,map_location='cpu',weights_only=True)
    assert state['policy_version']==request['policy_version'] and len(state['weights'])==80
    actor_hashes={n:hashlib.sha256(v.contiguous().numpy().tobytes()).hexdigest() for n,v in state['weights'].items()}
    old=native()
    study,cases,rollout=load('termination_first_independent_runtime',ROOT/'runtime.py').prepare()
    controller=rollout.__globals__['old'].c.controller(lambda m:'{}',study)
    by={c['source_id']:c for c in cases}
    interface=load('termination_first_independent_interface',N100/'interface.py')
    calls=0
    stages={}
    evidence=[]
    generation_failures=[]
    adapter_failures=[]
    failed_outcomes=[]
    for row in completion['rows']:
        folder=ex/'runs'/row['slot_id']
        session=interface.RecoverySession()
        records=[]
        current_generation_errors=0
        for p in sorted(folder.glob('call-*/returned.json')):
            record=json.loads(p.read_bytes())
            actual=json.loads((p.parent/'model-returned.json').read_bytes())
            validate_actual_input(session.rewrite(record['messages']),actual,old.raw_bytes)
            assert actual['arm']=='N' and actual['model_identity']['old_adapter_loaded'] is False
            assert actual['model_identity']['new_adapter']['adapter_model.safetensors']==ADAPTER_SHA
            assert actual['model_identity']['active_actor_checkpoint_state_sha256']==request['checkpoint_sha256']
            status=generation_status(record,actual)
            if status=='recorded_generation_error':
                current_generation_errors+=1
                generation_failures.append(dict(slot_id=row['slot_id'],call=p.parent.name,stage=actual['stage'],error=actual['error'],termination_reason=actual.get('termination_reason'),output_tokens=len(actual.get('output_token_ids') or [])))
            elif status=='recorded_adapter_error':
                adapter_failures.append(dict(slot_id=row['slot_id'],call=p.parent.name,stage=actual['stage'],error=record['error']))
            assert actual['route_enter']['effective_route']==('N' if actual['stage']=='Planner' else 'base_only')
            assert len(actual['route_enter']['modules'])==40 and len(actual['route_enter']['tensors_sha256'])==80
            for key in ('route_before','route_enter','route_after_forward','route_exit'):
                route=actual[key];expected='base_only' if key in ('route_enter','route_after_forward') and actual['stage']!='Planner' else 'N'
                assert route['effective_route']==expected and route['tensors_sha256']==actor_hashes and len(route['modules'])==40
                assert all(m['disabled']==(expected=='base_only') and m['active_adapters']==['default'] for m in route['modules'].values())
            opts=actual['generation_options'];assert opts['do_sample'] is False and opts['max_new_tokens']==512 and opts['max_time']<=90 and opts['eos_token_id']==[248044,248046]
            assert hashlib.sha256(old.raw_bytes(actual['output_token_ids'])).hexdigest()==actual['output_token_ids_sha256']
            assert hashlib.sha256(actual['raw_text'].encode('utf-8')).hexdigest()==actual['output_text_sha256']
            records.append(record)
            calls+=1
            stages[actual['stage']]=stages.get(actual['stage'],0)+1
        expected=json.loads((folder/'trace.json').read_bytes())
        replay=old.CPUReplay(records)
        case=by[row['source_id']]
        trace=rollout(case['item'],study,replay,case['snapshot'])
        assert same_trace(trace,expected) and replay.position==len(records)
        assert trace['new_steps']<=32
        outcome=outcome_status(trace,row,current_generation_errors)
        if outcome=='generation_failure_preserved':failed_outcomes.append(dict(slot_id=row['slot_id'],stop_reason=trace['stop_reason'],generation_errors=current_generation_errors))
        if trace['final_result'] is not None:
            assert controller.bridge.guard._valid_terminal(trace['final_result'],trace['state'])
        elif trace['terminated']:
            assert trace['stop_reason']=='abstained_terminal'
        if trace['stop_reason']=='best_effort_terminal':
            assert trace['terminal_event']['observation']['status']=='success'
            assert trace['terminal_event']['observation']['data']['final_result']==trace['final_result']
            assert not trace['termination']['evidence_sufficient'] and trace['termination']['requires_review']
        evidence.append(dict(slot_id=row['slot_id'],calls=len(records),native_replay_exact=True,
                             native_guard_valid=trace['final_result'] is not None,stop_reason=trace['stop_reason']))
    assert calls==completion['model_calls'] and calls<=17724
    claims=list((ex/'campaign/call-claims').glob('*.json'))
    assert len(claims)==calls
    result=dict(passed=True,new_native_replays=422,new_actual_model_calls=calls,
                new_results=422,reused_results=0,all_EOS=not generation_failures,all_input_rewrites_exact=True,
                generation_failures=generation_failures,adapter_failures=adapter_failures,failed_outcomes=failed_outcomes,
                audit_includes_recorded_failures=True,all_rows_processed=True,
                native_guard_checked=True,N_only=True,stages=stages,
                adapter_sha256=ADAPTER_SHA,actor_checkpoint_sha256=request['checkpoint_sha256'],archive_sha256=receipt['archive_sha256'],
                evidence=evidence,semantic_quality_verified_all=False)
    put(ROOT/'reviews/independent-verification.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='evidence'}))


if __name__=='__main__':
    main()
