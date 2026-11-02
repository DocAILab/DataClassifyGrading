"""One-shot fixed old failure packets, no rollout, training, or output repair."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parent
INPUTS_PATH=ROOT.parent/'finance_program_skill_context_contract_20261021_r1/inputs.json'
sys.path.insert(0,str(ROOT))
E2=ROOT.parent/'finance_program_student_b80_eval_20261021_r2'
sys.path.insert(1,str(E2))
from backend import backend_class
from native_checks import validate_raw
from prepare_runtime import exclusive_json

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def preflight():
    rows=json.loads(INPUTS_PATH.read_bytes())
    assert rows['logical_denominator']==15 and rows['unique_denominator']==14
    cls=backend_class(E2)
    assert 'validate_context_size' in cls.generate.__code__.co_names
    old=json.loads((ROOT.parent/'finance_program_skill_recovery20_20261021_r4/admission.json').read_bytes())
    s=importlib.util.spec_from_file_location('fixed_original_tokenizer',Path(old['training_root'])/'train_student.py')
    loader=importlib.util.module_from_spec(s);s.loader.exec_module(loader)
    tokenizer=loader._load_tokenizer(old['base_model_dir'])
    sizes=[]
    for r in rows['unique']:
        prompt=tokenizer.apply_chat_template(r['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        ids=tokenizer(prompt,add_special_tokens=False)['input_ids']
        assert ids and len(ids)+512<=49152
        sizes.append(dict(id=r['id'],input_tokens=len(ids),ids_sha256=hashlib.sha256(json.dumps(ids,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()))
    for value in [[],{},None,'']:
        try:validate_raw('INTERPRET_CONTEXT_FIELDS',json.dumps(dict(summary='observed',data=dict(context_field_semantics=value))))
        except Exception:pass
        else:raise ValueError('native validator relaxed')
    validate_raw('INTERPRET_CONTEXT_FIELDS',json.dumps(dict(summary='observed',data=dict(context_field_semantics='Observed fields; uncertainty retained.'))))
    return dict(unique=14,logical=15,sizes=sizes,context_tokens=49152,generation_allowance=512,no_truncation=True,new_model_calls=0,backend_compiled_guard=True)

def main():
    from launch_smoke import verify
    verify()
    admission=json.loads((ROOT/'admission.json').read_bytes())
    inputs=json.loads(INPUTS_PATH.read_bytes())
    folder=ROOT/'runs';folder.mkdir(exist_ok=False)
    exclusive_json(folder/'started.json',dict(pid=os.getpid(),time_unix=time.time(),inputs_sha256=sha(INPUTS_PATH)))
    backend=None;results=[]
    try:
        backend=backend_class(E2)(E2,admission)
        backend._ensure_loaded();backend._select('B')
        exclusive_json(folder/'loaded.json',dict(identity=backend.identity,time_unix=time.time()))
        for row in inputs['unique']:
            if row['stage']!='Planner':continue
            if time.time()>=admission['gpu_deadline_unix']-20:raise TimeoutError('absolute stage deadline')
            dest=folder/row['id'];dest.mkdir(exist_ok=False)
            exclusive_json(dest/'input.json',row)
            result=dict(id=row['id'],stage=row['stage'],original_paths=row['original_paths'],native_valid=False,generation_success=False)
            try:
                raw=backend.generate(row['messages'],'B',row['stage'],dest)
                result['generation_success']=True
                try:result.update(validate_raw(row['stage'],raw,row['native_state'],row['max_program_steps']))
                except Exception as exc:result.update(native_error_type=type(exc).__name__,native_error=str(exc))
            except Exception as exc:
                result.update(generation_error_type=type(exc).__name__,generation_error=str(exc))
                # Ordinary bounded output failures retain their denominator and
                # move forward; OOM/engine/storage/deadline never blindly retry.
                if 'generation did not finish with an actual EOS' not in str(exc):
                    exclusive_json(dest/'result.json',result);results.append(result);raise
            exclusive_json(dest/'result.json',result);results.append(result)
        reused=[]
        previous=ROOT.parent/'finance_program_skill_context_contract_20261021_r3'
        for key in ['input-13','input-14']:
            record=json.loads((previous/'runs'/key/'result.json').read_bytes())
            assert record['native_valid'] and record['generation_success']
            reused.append(dict(record,reused_from=str(previous/'runs'/key/'result.json')))
        all_results=results+reused
        assert len(results)==12 and len(all_results)==14
        by_id={r['id']:r for r in all_results}
        logical=[dict(original_path=r['original_path'],unique_input=r['unique_input'],
                      generation_success=by_id[r['unique_input']]['generation_success'],native_valid=by_id[r['unique_input']]['native_valid']) for r in inputs['logical']]
        exclusive_json(folder/'completion.json',dict(time_unix=time.time(),results=all_results,new_results=results,reused_results=reused,new_generation_denominator=12,logical=logical,unique_denominator=14,logical_denominator=15,training_updates=0,new_teacher_api_calls=0,
            all_old_overflows_generate=all(r['generation_success'] for r in results if r['stage']=='Planner'),
            all_context_native_valid=all(r['native_valid'] for r in all_results if r['stage']!='Planner'),
            scope='fixed-input generation and unchanged native static validation, not end-to-end correctness'))
        return 0
    except BaseException as exc:
        exclusive_json(folder/'failure.json',dict(error_type=type(exc).__name__,error=str(exc),time_unix=time.time(),completed_results=results))
        raise
    finally:
        if backend is not None and backend.runtime is not None:
            torch=backend.runtime['torch'];backend.runtime=None;backend=None
            import gc;gc.collect();torch.cuda.empty_cache()

if __name__=='__main__':
    if sys.argv[1:]==['--preflight']:print(json.dumps(preflight()))
    else:raise SystemExit(main())
