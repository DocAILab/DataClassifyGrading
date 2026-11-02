"""Unique bounded fresh162 training followed automatically by fixed evaluation."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from contract import ROOT,put,sha,validate_training_counts
from resource_profile import candidates,recheck
PY='/home/zhubh/finance_program_grpo_full_migrated_20261027_r3/dependencies/finance_program_5090_smoke_20261010/env/bin/python'
DEPS='/home/zhubh/finance_program_grpo_full_migrated_20261027_r3/dependencies/finance_program_adapter_compare_20261012_r2/deps'
LOCK_ROOT=Path('/home/zhubh/finance_program_grpo_full_migrated_20261027_r3/dependencies/finance_program_5090_smoke_20261010')
PARENT=ROOT.parent/'finance_program_student_fresh162_20261022_r3'
def verify():
    lock=json.loads((ROOT/'freeze.json').read_bytes())
    for v in lock['files']:
        p=ROOT/v['path']
        if not p.is_file() or p.is_symlink() or p.stat().st_size!=v['bytes'] or sha(p)!=v['sha256']:raise ValueError('frozen own file mismatch: '+v['path'])
    for v in lock['dependency_files']:
        p=ROOT.parent/v['path']
        if not p.is_file() or p.is_symlink() or sha(p)!=v['sha256']:raise ValueError('frozen dependency mismatch: '+v['path'])
    return lock
def snapshot():
    def query(args):return subprocess.check_output(['nvidia-smi']+args+['--format=csv,noheader,nounits'],text=True,timeout=30)
    proc={}
    for line in query(['--query-compute-apps=gpu_uuid,pid,used_memory']).splitlines():
        u,p,_=[x.strip() for x in line.split(',')];proc.setdefault(u,[]).append(int(p))
    rows=[]
    for line in query(['--query-gpu=index,uuid,memory.total,memory.free,compute_cap']).splitlines():
        i,u,t,f,c=[x.strip() for x in line.split(',')]
        rows.append(dict(index=int(i),uuid=u,total_bytes=int(t)*1024**2,free_bytes=int(f)*1024**2,bf16_compatible=int(c.split('.')[0])>=8,other_compute_pids=proc.get(u,[])))
    return rows
def lock_combo(combo):
    handles=[]
    try:
        for i in combo['gpu_indices']:
            h=(LOCK_ROOT/f'gpu{i}-experiment.lock').open('a+');handles.append(h);fcntl.flock(h.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        return handles
    except BlockingIOError:
        for h in handles:h.close()
        return None
def require_training_complete():
    d=json.loads((ROOT/'runs/training/completion.json').read_bytes())
    validate_training_counts(d)
    for n,h in d['final_adapter']['files_sha256'].items():
        if sha(ROOT/'runs/training/final-adapter'/n)!=h:raise ValueError('final new adapter mismatch')
def worker(stage,admission,handles,run):
    seconds=int(admission['gpu_deadline_unix']-time.time())-20
    if seconds<=0:raise TimeoutError('shared absolute GPU deadline exhausted')
    timeout=shutil.which('timeout')
    if not timeout:raise RuntimeError('GNU timeout required')
    command=[timeout,'--signal=TERM','--kill-after=20s',str(seconds)+'s',PY,'-B',str(ROOT/(stage+'.py'))]
    env=dict(os.environ);env.update(CUDA_VISIBLE_DEVICES=','.join(admission['gpu_uuids']),PYTHONPATH=DEPS,PYTHONUTF8='1',TOKENIZERS_PARALLELISM='false')
    with (run/(stage+'.stdout.log')).open('xb') as out,(run/(stage+'.stderr.log')).open('xb') as err:
        p=subprocess.Popen(command,cwd=ROOT,env=env,stdout=out,stderr=err,start_new_session=True,pass_fds=tuple(h.fileno() for h in handles))
        put(run/(stage+'-start.json'),dict(pid=p.pid,command=command,time_unix=time.time(),inherited_locks=len(handles)))
        code=p.wait()
    put(run/(stage+'-exit.json'),dict(actual_returncode=code,pid=p.pid,time_unix=time.time()))
    if code!=0:raise RuntimeError(stage+' actual exit '+str(code)+'; no blind retry')
def main():
    verify();run=ROOT/'supervision';run.mkdir(exist_ok=False);handles=[]
    run_lock=(ROOT/'writer.lock').open('a+');fcntl.flock(run_lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    put(run/'started.json',dict(pid=os.getpid(),time_unix=time.time(),gpu_started=False))
    try:
        proof=json.loads((ROOT/'reviews/tokenizer-audit-01.json').read_bytes())
        if not proof['all_fit'] or proof['rows']!=162 or proof['train_sha256']!=sha(ROOT/'data/train162.jsonl'):raise ValueError('actual full162 token proof required')
        origin=json.loads((PARENT/'admission.json').read_bytes())
        if json.loads((PARENT/'supervision/train-exit.json').read_bytes())['actual_returncode']!=1:raise ValueError('parent exit must remain1')
        wait_until=origin['gpu_deadline_unix']-60;admitted=None
        while time.time()<wait_until:
            for p in Path('/proc').iterdir():
                if not p.name.isdigit():continue
                try:
                    cmd=(p/'cmdline').read_bytes();state=(p/'stat').read_text().split(') ',1)[1].split()[0]
                    prior_roots=[PARENT,ROOT.parent/'finance_program_student_fresh162_20261022_r4']
                    if any(str(prior).encode() in cmd for prior in prior_roots) and b'-c' not in cmd and state!='Z':raise RuntimeError('prior writer remains live')
                except (FileNotFoundError,ProcessLookupError):pass
            from runtime_safety import guard_resources
            try:guard_resources(ROOT,preload=True)
            except RuntimeError:
                time.sleep(30);continue
            for combo in candidates(snapshot()):
                handles=lock_combo(combo)
                if handles is None:handles=[];continue
                try:admitted=recheck(snapshot(),combo)
                except ValueError:
                    for h in handles:h.close()
                    handles=[];continue
                break
            if admitted is not None:break
            time.sleep(30)
        if admitted is None:raise TimeoutError('original bounded window resource wait exhausted')
        admission=dict(admitted,gpu_started_unix=origin['gpu_started_unix'],gpu_deadline_unix=origin['gpu_deadline_unix'],resume_admitted_unix=time.time(),
            base_model_dir='/home/zhubh/finance_program_grpo_full_migrated_20261027_r3/dependencies/finance_program_5090_smoke_20261010/models/Qwen3.5-9B',max_sequence_tokens=49152,
            scientific_updates_allowed=11,token_audit_sha256=sha(ROOT/'reviews/tokenizer-audit-01.json'))
        put(ROOT/'admission.json',admission)
        worker('train',admission,handles,run);require_training_complete();verify()
        worker('evaluate',admission,handles,run)
        result=json.loads((ROOT/'runs/evaluation/completion.json').read_bytes())
        if len(result['rows'])!=80 or result['source_denominator']!=40:raise ValueError('fixed80 result slots required')
        put(run/'completion.json',dict(actual_returncode=0,time_unix=time.time(),training_rows=162,updates=11,evaluation_slots=80))
        return 0
    except BaseException as exc:
        put(run/'failure.json',dict(error_type=type(exc).__name__,error=str(exc),time_unix=time.time(),retry_allowed=False));raise
    finally:
        for h in handles:h.close()
        run_lock.close()
def watch():
    folder=ROOT/'launch';folder.mkdir(exist_ok=False)
    with (folder/'stdout.log').open('xb') as out,(folder/'stderr.log').open('xb') as err:
        p=subprocess.Popen([PY,'-B',str(ROOT/'supervisor.py')],cwd=ROOT,stdout=out,stderr=err)
        put(folder/'started.json',dict(watcher_pid=os.getpid(),supervisor_pid=p.pid,time_unix=time.time()))
        code=p.wait()
    put(folder/'exit.json',dict(actual_returncode=code,time_unix=time.time()));return code
if __name__=='__main__':raise SystemExit(watch() if sys.argv[1:]==['--watch'] else main())
