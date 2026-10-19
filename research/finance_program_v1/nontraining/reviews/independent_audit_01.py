"""Independent CPU-only audit: recorded model I/O + exact frozen trace replay.

Run only after the coordinator has closed all writers and verified/extracted the
new immutable snapshot. No torch/transformers import, GPU, network or model load.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SIGNATURE = '94ae314646625e078d7b0f8ee6315bf37778e1ce81bd8435a1417f43c769da1d'
REVISION = 'c202236235762e1c871ad0ccb60c8ee5ba337b9a'
BUDGET = dict(max_programs=3, max_program_steps=10, max_skill_steps=12, max_planner_attempts=4, max_identical_calls=1)
RUN_MODE = 'caller supplied backend I/O + real frozen executor; backend evidence in calls/'

class AuditError(AssertionError):
    pass

def require(value, message):
    if not value:
        raise AuditError(message)

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def exact(actual, expected, label):
    require(actual == expected, f'{label} mismatch')

class RecordedBackend:
    """Replays every external generation call, including retries and faults."""
    def __init__(self, records):
        self.records = records
        self.position = 0
        self.roles = []
        self.audit_faults = []

    def __call__(self, messages):
        try:
            require(self.position < len(self.records), 'replay requested missing call')
            record = self.records[self.position]
            self.position += 1
            exact(record['request']['messages'], messages, f'call {self.position} request messages')
            exact(record['result']['messages'], messages, f'call {self.position} result messages')
            self.roles.append('planner' if 'Global Program Planner' in messages[0]['content'] else 'skill')
            result = record['result']
            if result.get('error'):
                kind, separator, message = result['error'].partition(': ')
                require(separator and kind.isidentifier(), 'invalid recorded backend error')
                # Exact original class name and str(exc), without backend execution.
                raise type(kind, (Exception,), {})(message)
            return result['raw_text']
        except AuditError as exc:
            # Core deliberately catches generation errors. Retain audit failures
            # out of band so they cannot turn into apparently legitimate traces.
            self.audit_faults.append(str(exc))
            raise

    def finish(self):
        require(not self.audit_faults, '; '.join(self.audit_faults))
        require(self.position == len(self.records), 'unconsumed recorded model calls')

def replay_trace(saved, backend, registry, rows, budget, row):
    from program_loop import build_program_controller, ProgramConfig
    require(saved['execution_mode'] in (RUN_MODE, 'caller supplied model I/O + real executor'), 'trace execution mode invalid')
    controller = build_program_controller(backend, registry, rows, ProgramConfig(**budget))
    trace = controller.run({'domain':row['domain'], 'field_name':row['metadata']['field_name'], 'source_id':row['id']})
    if saved['execution_mode'] == RUN_MODE:
        trace.execution_mode = RUN_MODE
    require(not backend.audit_faults, '; '.join(backend.audit_faults))
    actual = trace.to_dict()
    # Includes parser raw, every message, full observations/feedback, all binding
    # metadata/check-dependencies, invalidations, availability, lineage, budgets,
    # state, stop reason and terminal result. No outcome-specific shortcut.
    exact(actual, saved, 'complete trace replay')
    first = json.loads(saved['planner_attempts'][0]['messages'][1]['content'])
    require(row['metadata']['field_description'] not in json.dumps(first,ensure_ascii=False), 'initial metadata description exposed')
    require(len(first['skill_library']) == 14, 'P14 library incomplete')
    for attempt in saved['planner_attempts']:
        combined = json.dumps(attempt['messages'],ensure_ascii=False)
        require(all(r['id'] not in combined for r in rows), 'source ID exposed in planner messages')
    for program in saved['programs']:
        for step in program['steps']:
            if step['call']['tool'] == 'GET_FIELD_METADATA' and step['status'] == 'success':
                exact(step['data']['field_metadata'],row['metadata'],'actual metadata acquisition')
    return actual

def raw_steps(raw):
    try:
        parsed = json.loads(raw)
        steps = parsed.get('program') if isinstance(parsed,dict) else None
        return steps if isinstance(steps,list) else []
    except (TypeError,ValueError):
        return []

def trace_summary(traces):
    attempts, errors, execution = Counter(), Counter(), Counter()
    details = []
    for trace in traces:
        row_attempts = []
        for attempt in trace['planner_attempts']:
            steps = raw_steps(attempt['raw_response'])
            attempts[attempt['status']] += 1
            if 'error' in attempt:
                errors[(attempt['status'],attempt['error'])] += 1
            row_attempts.append({'attempt':attempt['attempt'],'status':attempt['status'],'raw_steps':len(steps),'tools':[s.get('tool') if isinstance(s,dict) else None for s in steps],'error':attempt.get('error')})
        steps = [s for p in trace['programs'] for s in p['steps']]
        for step in steps:
            execution[(step['call']['tool'],step['status'])] += 1
        details.append({'input':trace['input'],'attempts':row_attempts,'budget':trace['budget'],
            'raw_multistep_programs':sum(len(raw_steps(a['raw_response']))>1 for a in trace['planner_attempts']),
            'accepted_multistep_programs':sum(len(p['plan']['program'])>1 for p in trace['programs']),
            'executed_multistep_programs':sum(len(p['steps'])>1 for p in trace['programs']),
            'program_status':dict(Counter(p['status'] for p in trace['programs'])),
            'actual_skills':[{'tool':s['call']['tool'],'status':s['status'],'summary':s.get('summary')} for s in steps],
            'terminated':trace['terminated'],'stop_reason':trace['stop_reason'],'final_result':trace['final_result']})
    return {'metric_scope':'unlabelled engineering smoke; no accuracy metric','samples':len(traces),
        'planner_attempts':sum(attempts.values()),'accepted_plans':sum(len(t['programs']) for t in traces),
        'attempt_status':dict(attempts),'planner_errors':[{'status':status,'reason':error,'count':count} for (status,error),count in errors.items()],
        'execution':[{'tool':tool,'status':status,'count':count} for (tool,status),count in execution.items()],
        'skill_steps':sum(execution.values()),'legal_terminations':sum(t['terminated'] for t in traces),
        **{key:sum(d[key] for d in details) for key in ('raw_multistep_programs','accepted_multistep_programs','executed_multistep_programs')},'details':details}

def call_summary(records, roles):
    require(len(records)==len(roles),'model call role denominator mismatch')
    peaks = [r['memory'].get('max_memory_allocated') for r in records]
    return {'model_calls':len(records),'planner_calls':roles.count('planner'),'skill_model_calls':roles.count('skill'),
        'input_tokens':sum(r.get('input_tokens') or 0 for r in records),'output_tokens':sum(r.get('output_tokens') or 0 for r in records),
        'generation_seconds':sum(r.get('seconds') or 0 for r in records),
        'peak_allocated_bytes':max((p for p in peaks if p is not None),default=None),
        'backend_errors':sum(bool(r.get('error')) for r in records),'telemetry_errors':sum(bool(r.get('telemetry_errors')) for r in records),
        'output_budget_truncations':sum(bool(r['truncation_at_budget']) for r in records),
        'unknown_input_token_calls':sum(r.get('input_tokens') is None for r in records),
        'unknown_output_token_calls':sum(r.get('output_tokens') is None for r in records),
        'unknown_peak_vram_calls':sum(p is None for p in peaks)}

def validate_call(request,result,backend):
    exact(request['generation'],backend['generation'],'request generation')
    exact(result['generation'],backend['generation'],'result generation')
    exact(result['parameter_devices'],['cuda:0'],'parameter devices')
    exact(result['parameter_dtypes'],['torch.float16'],'parameter dtypes')
    exact(result['attention_implementation'],backend['attention_implementation'],'attention implementation')
    require(type(result['seconds']) in (int,float) and math.isfinite(result['seconds']) and result['seconds']>=0,'generation duration invalid')
    error = result['error']
    for key in ('input_tokens','output_tokens'):
        value = result[key]
        require(value is None or type(value) is int and value>=0, f'{key} invalid')
    if not error:
        require(isinstance(result['raw_text'],str),'successful generation raw text missing')
        exact(result['input_devices'],['cuda:0'],'input devices')
        require(result['input_tokens'] is not None and result['input_tokens']+512<=32768,'context cap violated')
        require(result['output_tokens'] is not None and result['output_tokens']<=512,'output cap violated')
        exact(result['truncation_at_budget'],result['output_tokens']>=512,'output budget flag')
    else:
        require(result['raw_text'] is None,'backend fault with raw text')
        if result['output_tokens'] is None:
            require(not result['truncation_at_budget'],'backend fault output evidence invalid')
        else:
            # Backend records tokens/flag before decode, which can itself fail.
            exact(result['input_devices'],['cuda:0'],'input devices')
            require(result['input_tokens'] is not None and result['input_tokens']+512<=32768,'context cap violated')
            require(result['output_tokens']<=512,'output cap violated')
            exact(result['truncation_at_budget'],result['output_tokens']>=512,'output budget flag')
        if error.startswith('ContextBudgetExceeded:'):
            require(result['input_tokens'] is not None and result['input_tokens']+512>32768,'context fault below cap')
            exact(result['input_devices'],[],'context failure before tensor transfer')
    exact(set(result['memory']),{'memory_allocated','memory_reserved','max_memory_allocated'},'memory telemetry keys')
    require(set(result['telemetry_errors'])<=set(result['memory']),'unknown telemetry error')
    for name,value in result['memory'].items():
        require((value is None and name in result['telemetry_errors']) or (type(value) is int and value>=0 and name not in result['telemetry_errors']), 'memory metric/error mismatch')

def validate_sources(snapshot,environment):
    manifest = read(ROOT/'reviews/source-manifest-01.json')
    entries = manifest['files']
    require(len(entries)==121,'frozen source count mismatch')
    signature = hashlib.sha256(''.join(f"{e['path']}\t{e['bytes']}\t{e['sha256']}\n" for e in sorted(entries,key=lambda e:e['path'])).encode()).hexdigest()
    exact(signature,SIGNATURE,'frozen source signature')
    for entry in entries:
        for base in (ROOT,snapshot):
            path = base/entry['path']
            require(path.is_file() and path.stat().st_size==entry['bytes'] and sha(path)==entry['sha256'],f'frozen source hash mismatch: {path}')
    core = {e['path']:e['sha256'] for e in entries if e['path'].split('/')[0] in ('program_loop','vendor','tools')}
    require(len(core)==109,'core manifest count must be exactly 109')
    actual = {p.relative_to(snapshot).as_posix() for base in ('program_loop','vendor','tools') for p in (snapshot/base).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    exact(actual,set(core),'core source inventory')
    exact(environment['core_source_sha256'],core,'environment all 109 core SHA')

def audit_snapshot(snapshot,reference_inputs,model_manifest):
    snapshot = Path(snapshot).resolve()
    run = snapshot/'runs/smoke-01'
    inputs = read(run/'inputs.json')
    exact(inputs,read(reference_inputs),'same exact historical unlabelled inputs/registry/budget')
    exact(inputs['budget'],BUDGET,'fixed smoke budget')
    rows,registry = inputs['rows'],inputs['registry']
    require(len(rows)==3 and len(registry)==6,'three rows and six registry entries required')
    require(len({r['id'] for r in rows})==3,'source IDs not unique')
    require(all(set(r)=={'id','domain','metadata'} and set(r['metadata'])=={'field_name','field_description'} for r in rows),'labels or unexpected initial fields')
    env = read(run/'environment.json')
    validate_sources(snapshot,env)
    expected_env = {'seed':20260928,'context_cap':32768,'max_new_tokens':512,'per_generate_max_time':90.0,'whole_process_timeout_seconds':900,
                    'whole_process_timeout_enforced_by':'external coordinator','inference':'FP16 cuda:0 batch1 greedy enable_thinking=False'}
    for key,value in expected_env.items():
        exact(env[key],value,f'environment {key}')
    require(type(env['pid']) is int and env['pid']>0,'environment PID invalid')
    require(all(env['packages'].get(p) for p in ('torch','transformers','accelerate','huggingface_hub','safetensors')),'environment package versions missing')
    trusted = read(ROOT/'reviews/trusted-model-manifest-01.json')
    exact(read(model_manifest),trusted,'trusted model manifest identity/files/bytes/SHA')
    require(trusted['repo_id']=='Qwen/Qwen3.5-9B' and trusted['revision']==REVISION and len(trusted['files'])==11 and trusted['total_bytes']==19329302129,'official pinned identity invalid')
    backend = read(run/'calls/backend.json')
    for key,value in {'repo_id':'Qwen/Qwen3.5-9B','revision':REVISION,'parameter_devices':['cuda:0'],'parameter_dtypes':['torch.float16'],
                      'offline_options':{'local_files_only':True,'trust_remote_code':False},'config_type':'qwen3_5','model_class':'Qwen3_5ForConditionalGeneration'}.items():
        exact(backend[key],value,f'backend {key}')
    gen = backend['generation']
    exact(gen,{'do_sample':False,'max_new_tokens':512,'max_time':90.0,'eos_token_id':248044,'pad_token_id':248044},'fixed generation/EOS')
    paths = sorted((run/'calls').glob('call-*'))
    exact([p.name for p in paths],[f'call-{i:04d}' for i in range(1,len(paths)+1)],'contiguous model call records')
    records = []
    for path in paths:
        request,result = read(path/'request.json'),read(path/'result.json')
        validate_call(request,result,backend)
        combined = json.dumps(request['messages'],ensure_ascii=False)
        require(all(row['id'] not in combined for row in rows),'source ID exposed in model messages')
        records.append({'request':request,'result':result})
    # Import only after validating every source byte; process must not have loaded
    # another copy of program_loop before this point.
    require('program_loop' not in sys.modules,'audit CLI must run in a fresh process')
    sys.path.insert(0,str(snapshot))
    replay = RecordedBackend(records)
    traces, row_calls = [], []
    for row in rows:
        start = replay.position
        traces.append(replay_trace(read(run/'samples'/(row['id']+'.json')),replay,registry,rows,inputs['budget'],row))
        row_calls.append(call_summary([r['result'] for r in records[start:replay.position]],replay.roles[start:replay.position]))
    replay.finish()
    require('torch' not in sys.modules and 'transformers' not in sys.modules,'CPU audit imported model runtime')
    report = {**trace_summary(traces),**call_summary([r['result'] for r in records],replay.roles),
        'source_signature_sha256':SIGNATURE,'core_files_sha_matches':109,'model_revision':REVISION,
        'raw_request_trace_exact_match':True,'full_trace_replay_exact_match':True,'normalized_bindings_dependencies_verified':True,
        'initial_metadata_description_hidden':True,'source_ids_not_in_planner_messages':True,'same_unlabelled_inputs_as_previous_smoke':True,
        'all_source_ids_not_in_model_messages':True,
        'snapshot':str(snapshot),'reference_inputs_sha256':sha(reference_inputs),'trusted_model_manifest_sha256':sha(ROOT/'reviews/trusted-model-manifest-01.json')}
    summary = read(run/'summary.json')
    for key in ('model_calls','planner_attempts','accepted_plans','skill_steps','input_tokens','output_tokens','generation_seconds'):
        exact(summary[key],report[key],f'runner summary {key}')
    exact(summary['terminated_count'],report['legal_terminations'],'runner summary legal terminations')
    expected_errors = [{'error':r['result']['error']} for r in records if r['result']['error']]
    exact(summary['backend_errors'],expected_errors,'all backend failure denominator')
    exact(summary['status'],'backend_failure' if expected_errors else 'completed','runner completion status')
    expected_samples = [{'id':row['id'],'terminated':t['terminated'],'stop_reason':t['stop_reason'],'budget':t['budget'],
        'planner_errors':sum('error' in a for a in t['planner_attempts']),
        'execution_errors':sum(s['status']!='success' for p in t['programs'] for s in p['steps'])} for row,t in zip(rows,traces)]
    exact(summary['samples'],expected_samples,'all runner samples including failures')
    for row,detail,calls in zip(rows,report['details'],row_calls):
        detail['id'] = row['id']
        detail.update(calls)
        detail['execution_errors'] = sum(s['status']!='success' for s in detail['actual_skills'])
    report['call_details'] = [{'call':p.name,'role':role,**{key:r['result'][key] for key in ('input_tokens','output_tokens','seconds','memory','telemetry_errors','error','truncation_at_budget')}} for p,r,role in zip(paths,records,replay.roles)]
    return report

def write_report(path,report):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot',type=Path,default=ROOT/'remote-snapshot-01')
    parser.add_argument('--reference-inputs',type=Path,default=ROOT/'reviews/reference-inputs.json')
    parser.add_argument('--model-manifest',type=Path)
    args = parser.parse_args(argv)
    target = ROOT/'reviews/independent-smoke-audit-01.json'
    require(not target.exists(),'exclusive audit output already exists')
    report = audit_snapshot(args.snapshot,args.reference_inputs,args.model_manifest or args.snapshot/'reviews/remote-model-manifest-01.json')
    write_report(target,report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('details','call_details')},ensure_ascii=True,indent=2))

if __name__ == '__main__':
    main()
