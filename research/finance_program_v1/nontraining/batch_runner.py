"""One exclusive cohort with the frozen Program controller and Backend."""
import argparse
from pathlib import Path
import time
from local_smoke import Backend, environment, reserve_run
from prepare_batch import ROOT, BUDGET, read, save, sha, validate_study

RUN_MODE='caller supplied backend I/O + real frozen executor; backend evidence in calls/'

def verify_study_identity(study_path):
    path=Path(study_path)
    if read(path.with_name('study-manifest.json'))['study_sha256']!=sha(path): raise ValueError('prepared study identity changed')
    return True

def run_batch(model,output,cohort,*,study_path=ROOT/'data/study.json',backend_factory=Backend):
    verify_study_identity(study_path)
    study=validate_study(read(study_path))
    output=reserve_run(output); started=time.perf_counter()
    ids=study['cohorts'][cohort]; by_id={r['id']:r for r in study['rows']}
    summary={'status':'initialization_failure','cohort':cohort,'expected_ids':ids,'expected_count':len(ids),
             'study_sha256':sha(study_path),'terminated_count':0,'samples':[{'id':i,'status':'pending'} for i in ids]}
    try:
        env=environment(model); env['whole_process_timeout_seconds']=7200
        env['backend_kind']='production' if backend_factory is Backend else 'scripted_cpu_test'
        save(output/'environment.json',env)
        save(output/'inputs.json',{'cohort':cohort,'expected_ids':ids,'study_sha256':sha(study_path),'budget':BUDGET})
        backend=backend_factory(model,output/'calls')
        from program_loop import ProgramConfig,build_program_controller
        config=ProgramConfig(**BUDGET); summary['status']='completed'
        for entry in summary['samples']:
            print(f'{cohort} start {entry["id"]}',flush=True)
            row=by_id[entry['id']]; entry.update(status='error',call_start=len(backend.calls))
            controller=None
            try:
                controller=build_program_controller(backend,study['registry'],study['rows'],config)
                trace=controller.run({'source_id':row['id'],'domain':row['domain'],'field_name':row['metadata']['field_name']})
                trace.execution_mode=RUN_MODE; trace.write_json(output/'samples'/(row['id']+'.json'))
                entry.update(status='completed',terminated=trace.terminated,stop_reason=trace.stop_reason,budget=trace.budget)
                summary['terminated_count']+=int(trace.terminated)
            except Exception as exc:
                entry['error']=f'{type(exc).__name__}: {exc}'
                summary['status']='runner_failure'
                save(output/'errors'/(row['id']+'.json'),{'input':{'source_id':row['id'],'domain':row['domain'],'field_name':row['metadata']['field_name']},'error':entry['error']})
            finally:
                entry['call_end']=len(backend.calls)
                save(output/'ranges'/(row['id']+'.json'),entry)
                save(output/'progress'/f'{summary["samples"].index(entry)+1:04d}.json',summary)
                print(f'{cohort} done {entry["id"]} {entry["status"]} terminated={entry.get("terminated",False)}',flush=True)
        if backend.failures: summary['status']='backend_failure'
        summary['backend_errors']=backend.failures
    except Exception as exc:
        if summary['status']!='initialization_failure': summary['status']='runner_failure'
        summary['error']=f'{type(exc).__name__}: {exc}'
    finally:
        records=[]; record_errors=[]
        for path in sorted((output/'calls').glob('call-*')):
            try: records.append(read(path/'result.json'))
            except Exception as exc: record_errors.append({'call':path.name,'error':f'{type(exc).__name__}: {exc}'})
        if record_errors:
            summary['record_errors']=record_errors
            if summary['status']!='initialization_failure': summary['status']='runner_failure'
        summary.update(model_calls=len(records),input_tokens=sum(r.get('input_tokens') or 0 for r in records),
                       output_tokens=sum(r.get('output_tokens') or 0 for r in records),generation_seconds=sum(r.get('seconds') or 0 for r in records),seconds=time.perf_counter()-started)
        save(output/'summary.json',summary)
    return 0 if summary['status']=='completed' else 2

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--model',required=True,type=Path); p.add_argument('--output',required=True,type=Path)
    p.add_argument('--cohort',required=True,choices=('smoke','pilot','expanded')); p.add_argument('--study',type=Path,default=ROOT/'data/study.json')
    a=p.parse_args(argv); return run_batch(a.model,a.output,a.cohort,study_path=a.study)
if __name__=='__main__': raise SystemExit(main())
