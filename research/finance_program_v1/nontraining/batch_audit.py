"""CPU recorded-I/O replay and separate local gold evaluation."""
# 先用已记录的模型 I/O 重跑控制器并核对完整轨迹，再在独立评分入口读取私有答案。
import argparse
from collections import Counter
import json
from pathlib import Path
from prepare_batch import ROOT,BUDGET,read,save,sha,validate_study
from batch_runner import RUN_MODE,verify_study_identity
from reviews.independent_audit_01 import RecordedBackend,AuditError,require,exact,validate_sources,validate_call,REVISION

def audit(run,*,study_path=ROOT/'data/study.json',allow_scripted_fixture=False):
    verify_study_identity(study_path)
    run=Path(run); study=validate_study(read(study_path)); summary=read(run/'summary.json')
    env=read(run/'environment.json'); validate_sources(ROOT,env)
    for key,value in {'seed':20260928,'context_cap':32768,'max_new_tokens':512,'per_generate_max_time':90.0,
                      'whole_process_timeout_seconds':7200,'whole_process_timeout_enforced_by':'external coordinator',
                      'inference':'FP16 cuda:0 batch1 greedy enable_thinking=False'}.items(): exact(env[key],value,'environment '+key)
    exact(summary['study_sha256'],sha(study_path),'study identity')
    exact(summary['expected_ids'],study['cohorts'][summary['cohort']],'cohort identity')
    exact(summary['expected_count'],len(summary['expected_ids']),'full denominator')
    exact([s['id'] for s in summary['samples']],summary['expected_ids'],'sample denominator')
    exact(read(run/'inputs.json')['budget'],BUDGET,'fixed budget')
    paths=sorted((run/'calls').glob('call-*'))
    exact([p.name for p in paths],[f'call-{i:04d}' for i in range(1,len(paths)+1)],'continuous calls')
    records=[{'request':read(p/'request.json'),'result':read(p/'result.json')} for p in paths]
    if summary['status']=='initialization_failure':
        require(all(entry=={'id':ident,'status':'pending'} for entry,ident in zip(summary['samples'],summary['expected_ids'])),
                'initialization failure must retain only untouched pending samples')
        require(not records and summary['model_calls']==0,'initialization failure cannot contain model calls')
        require(isinstance(summary.get('error'),str) and bool(summary['error'].strip()),'initialization failure error record missing')
        for folder in ('ranges','samples','errors'):
            require(not any(p.is_file() for p in (run/folder).rglob('*')),'initialization failure has unexpected '+folder+' artifacts')
    require(env.get('backend_kind') in ('production','scripted_cpu_test'),'backend evidence kind missing')
    require(env['backend_kind']=='production' or allow_scripted_fixture is True,'formal audit requires production backend evidence')
    if env['backend_kind']=='production' and summary['status']!='initialization_failure':
        backend=read(run/'calls/backend.json')
        for key,value in {'repo_id':'Qwen/Qwen3.5-9B','revision':REVISION,'parameter_devices':['cuda:0'],'parameter_dtypes':['torch.float16'],
                          'offline_options':{'local_files_only':True,'trust_remote_code':False},'config_type':'qwen3_5','model_class':'Qwen3_5ForConditionalGeneration','attention_implementation':'sdpa',
                          'generation':{'do_sample':False,'max_new_tokens':512,'max_time':90.,'eos_token_id':248044,'pad_token_id':248044}}.items(): exact(backend[key],value,'production backend '+key)
        for record in records: validate_call(record['request'],record['result'],backend)
    for record in records:
        text=json.dumps(record['request']['messages'],ensure_ascii=False)
        require(all(row['id'] not in text for row in study['rows']),'private ID in model messages')
    from program_loop import ProgramConfig,build_program_controller
    by_id={r['id']:r for r in study['rows']}; position=0; verified=0; errors=[]
    for entry in summary['samples']:
        if entry['status']=='pending':
            require(summary['status']=='initialization_failure','pending after initialization'); continue
        exact(entry['call_start'],position,'sample call start'); end=entry['call_end']
        require(type(end) is int and position<=end<=len(records),'invalid call range')
        replay=RecordedBackend(records[position:end]); row=by_id[entry['id']]
        input_value={'source_id':row['id'],'domain':row['domain'],'field_name':row['metadata']['field_name']}
        try:
            controller=build_program_controller(replay,study['registry'],study['rows'],ProgramConfig(**BUDGET))
            trace=controller.run(input_value)
        except Exception as exc:
            require(not replay.audit_faults,'; '.join(replay.audit_faults))
            require(entry['status']=='error','unexpected independent controller exception')
            exact(f'{type(exc).__name__}: {exc}',entry['error'],'independent exception identity')
            exact(read(run/'errors'/(row['id']+'.json')),{'input':input_value,'error':entry['error']},'error artifact')
            errors.append({'id':row['id'],'independently_reproduced_error':entry['error']})
        else:
            require(entry['status']=='completed','saved exception did not reproduce')
            trace.execution_mode=RUN_MODE; actual=trace.to_dict()
            exact(actual,read(run/'samples'/(row['id']+'.json')),'complete actual trace replay')
            exact(entry['terminated'],trace.terminated,'termination summary')
            exact(entry['budget'],trace.budget,'budget summary')
            exact(entry['stop_reason'],trace.stop_reason,'stop reason summary')
            if trace.planner_attempts:
                first=json.loads(trace.planner_attempts[0]['messages'][1]['content'])
                require(len(first['skill_library'])==14,'P14 library changed')
                require('field_description' not in first.get('record',{}),'initial description exposed')
            from pool_metadata import FIELDS,present
            expected_metadata={k:v for k,v in row['metadata'].items() if k in FIELDS and present(v)}
            for program in actual['programs']:
                for step in program['steps']:
                    if step['call']['tool']=='GET_FIELD_METADATA' and step['status']=='success': exact(step['data']['field_metadata'],expected_metadata,'actual filtered metadata identity')
            verified+=1
        replay.finish(); position=end
        exact(entry,read(run/'ranges'/(row['id']+'.json')),'actual call range artifact')
    exact(position,len(records),'all calls consumed')
    exact(summary['model_calls'],len(records),'call summary')
    for key in ('input_tokens','output_tokens'):
        exact(summary[key],sum(r['result'].get(key) or 0 for r in records),'token summary '+key)
    exact(summary['generation_seconds'],sum(r['result'].get('seconds') or 0 for r in records),'generation wall time')
    exact(summary['terminated_count'],sum(bool(s.get('terminated')) for s in summary['samples']),'termination denominator')
    return {'backend_kind':env['backend_kind'],'verified_samples':verified,'independently_verified_errors':errors,'pending_count':sum(s['status']=='pending' for s in summary['samples']),
            'model_calls':len(records),'all_calls_consumed':True,'full_trace_exact':True}

def macro_f1(golds,predictions,labels):
    scores=[]
    for label in labels:
        tp=sum(g==label and p==label for g,p in zip(golds,predictions)); fp=sum(g!=label and p==label for g,p in zip(golds,predictions)); fn=sum(g==label and p!=label for g,p in zip(golds,predictions))
        scores.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
    return sum(scores)/len(scores) if scores else 0.

def evaluate_data(study,summary,traces,gold):
    ids=summary['expected_ids']; require(len(ids)==len(set(ids)),'duplicate evaluation IDs')
    require(set(ids)<=set(gold['answers']),'gold missing expected IDs')
    by_id={t['id']:t['trace'] for t in traces}; require(len(by_id)==len(traces),'duplicate traces')
    require(set(by_id)<=set(ids),'unexpected evaluation trace')
    counts=Counter(); skills=Counter(); successes=Counter(); unavailable=Counter(); labels=[]; predictions=[]
    for ident in ids:
        trace=by_id.get(ident,{}); terminated=bool(trace.get('terminated')); final=trace.get('final_result') or {}; answer=gold['answers'][ident]
        prediction=final.get('category') if terminated else None; level=final.get('grading_level') if terminated else None
        if terminated: require(prediction in study['registry'] and study['registry'][prediction]['grading_level']==level,'invalid terminal Registry binding')
        labels.append(answer['category']); predictions.append(prediction)
        counts['terminated']+=terminated; counts['category_correct']+=terminated and prediction==answer['category']; counts['grading_correct']+=terminated and level==answer['grading_level']; counts['joint_correct']+=terminated and prediction==answer['category'] and level==answer['grading_level']
        history=[]; recovered=False
        for program in trace.get('programs',[]):
            for step in program['steps']:
                tool=step['call']['tool']; skills[tool]+=1; successes[tool]+=step['status']=='success'; unavailable[tool]+=step['status']=='unavailable'
                recovered |= bool(step.get('controller_intervention'))
                if tool=='SELECT_CATEGORY_TOPK' and step['status']=='success':
                    candidates=step['data']['standard_candidates']; candidates=[c.get('category') or c.get('category_id') for c in candidates]
                    require(len(candidates)==5 and len(set(candidates))==5 and set(candidates)<=set(study['registry']),'invalid successful Top5')
                    history.append(candidates)
        counts['recovered_terminated']+=terminated and recovered; counts['natural_terminated']+=terminated and not recovered
        adaptive=len(trace.get('planner_attempts',[]))>1 or len(trace.get('programs',[]))>1
        counts['autonomous_adaptive_replan_terminated']+=terminated and not recovered and adaptive
        counts['top5_first_hit']+=bool(history and answer['category'] in history[0]); counts['top5_final_hit']+=bool(history and answer['category'] in history[-1]); counts['top5_any_hit']+=any(answer['category'] in h for h in history)
        counts['samples_with_top5']+=bool(history)
    for key in ('terminated','category_correct','grading_correct','joint_correct','natural_terminated','recovered_terminated','autonomous_adaptive_replan_terminated','top5_first_hit','top5_final_hit','top5_any_hit','samples_with_top5'): counts[key]+=0
    result={'denominator':len(ids),**dict(counts),'macro_f1_supported':macro_f1(labels,predictions,gold['supported_classes']),
            'macro_f1_fixed12':macro_f1(labels,predictions,sorted(study['registry'])),'supported_class_count':len(gold['supported_classes']),
            'skills_called':dict(skills),'skills_succeeded':dict(successes),'skills_unavailable':dict(unavailable),
            'runner_seconds':summary['seconds'],**{k:summary.get(k,0) for k in ('model_calls','input_tokens','output_tokens','generation_seconds')}}
    result['metric_definitions']={
        'recovered_terminated':'Controller intervention present in at least one executed step of a legally terminated trace.',
        'natural_terminated':'Autonomous legal termination with no controller intervention; includes model-directed adaptive replanning.',
        'autonomous_adaptive_replan_terminated':'Autonomous legal termination after more than one actual planner attempt or accepted program; a subset of natural_terminated.'}
    result.update({k+'_rate':counts[k]/len(ids) if ids else 0 for k in counts})
    return result

def evaluate(run,gold_path,*,study_path=ROOT/'data/study.json'):
    run=Path(run); audit(run,study_path=study_path)
    return evaluate_data(read(study_path),read(run/'summary.json'),[{'id':p.stem,'trace':read(p)} for p in sorted((run/'samples').glob('*.json'))],read(gold_path))

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--run',type=Path,required=True); p.add_argument('--study',type=Path,default=ROOT/'data/study.json'); p.add_argument('--gold',type=Path); p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(argv); result=evaluate(a.run,a.gold,study_path=a.study) if a.gold else audit(a.run,study_path=a.study); save(a.output,result); print(json.dumps(result,ensure_ascii=False)); return 0
if __name__=='__main__': raise SystemExit(main())
