"""Detached CPU watcher; no GPU allocations before completed training acceptance."""
import argparse
import datetime
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from io_utils import ROOT,read,put,sha,verify
from gate import training_ready,cpu_proof_identity


def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def live(pid,needle):
    p=Path('/proc')/str(pid)
    try:
        return bool(p.exists() and needle in (p/'cmdline').read_bytes().decode(errors='replace')
            and (p/'stat').read_text().split()[2]!='Z')
    except (FileNotFoundError,ProcessLookupError):return False


def event(kind,**value):
    folder=ROOT/'runs/events';folder.mkdir(parents=True,exist_ok=True)
    put(folder/('%05d.json'%(len(list(folder.glob('*.json')))+1)),{'kind':kind,'time':now(),**value})
    print(kind,now(),flush=True)


def parent_state(c):
    root=Path(c['training_root'])
    def optional(n):
        try:return read(root/n) if (root/n).is_file() else None
        except __import__('json').JSONDecodeError:return None  # a stage writer may still be flushing a new receipt
    worker=optional('runs/worker.json');formal=optional('runs/formal.pid.json')
    exits={k:optional('runs/'+k+'.exit.json') for k in ('trial','formal','pipeline','archive')}
    return {'manifest_sha256':sha(root/'manifest.json'),
        'writer_live':any(live(p['pid'],needle) for p,needle in ((worker,'pipeline.py'),(formal,'train_student.py')) if p),
        'exits':{k:v['exit_code'] if v else None for k,v in exits.items()},
        'completion':optional('formal/completion.json'),'cpu_proof':optional('reviews/post-training-cpu-proof.json'),
        'adapter_sha256':sha(Path(c['adapter'])/'adapter_model.safetensors') if (Path(c['adapter'])/'adapter_model.safetensors').is_file() else None}


def cpu_proof(c):
    state=parent_state(c);root=Path(c['training_root'])
    if state['manifest_sha256']!=c['training_manifest_sha256']:raise ValueError('parent signature mismatch')
    if any(v not in (0,None) for v in state['exits'].values()):raise RuntimeError('training stage failed; evaluation not authorized')
    if state['writer_live'] or any(v!=0 for v in state['exits'].values()):return False
    if not state['cpu_proof']:
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
        with (ROOT/'runs/training-cpu-verify.log').open('xb') as log:
            p=subprocess.run([c['python'],'-B',str(ROOT/'verify_training.py'),'--training-root',str(root)],cwd=ROOT,env=env,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        put(ROOT/'runs/training-cpu-verify.exit.json',{'exit_code':p.returncode,'time':now()})
        if p.returncode:raise RuntimeError('independent training CPU verification failed')
        state=parent_state(c)
    if not training_ready(state,c['training_manifest_sha256']):raise ValueError('final adapter/CPU proof gate failed')
    cfg=Path(c['adapter'])/'adapter_config.json'
    if not cpu_proof_identity(state['cpu_proof'],sha(cfg),state['completion']['adapter_config_sha256'],
            sha(ROOT/'verify_training.py'),sha(ROOT/'manifest.json')):
        raise ValueError('CPU proof config or full validator identity differs')
    put(ROOT/'runs/training-anchor.json',{'training_manifest_sha256':state['manifest_sha256'],
        'adapter_sha256':state['adapter_sha256'],'adapter_config_sha256':sha(cfg),
        'completion_sha256':sha(root/'formal/completion.json'),
        'cpu_proof_sha256':sha(root/'reviews/post-training-cpu-proof.json'),
        'parent_archive':read(root/'exports/final-01/archive.json'),'accepted_at':now()})
    # Preserve the exact newly trained bytes alongside evaluation evidence, not only an external path.
    for source,name in [(root/'manifest.json','manifest.json'),(root/'formal/completion.json','completion.json'),
            (root/'reviews/post-training-cpu-proof.json','cpu-proof.json'),
            (Path(c['adapter'])/'adapter_model.safetensors','adapter_model.safetensors'),
            (cfg,'adapter_config.json')]:
        target=ROOT/'evidence/training'/name;target.parent.mkdir(parents=True,exist_ok=True)
        with source.open('rb') as src,target.open('xb') as dst:__import__('shutil').copyfileobj(src,dst)
    return True


def resource(c):
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
    selected=[list(map(str.strip,line.split(','))) for line in rows.splitlines() if line.split(',')[0].strip()==c['gpu_index']]
    if len(selected)!=1 or selected[0][1]!=c['gpu_uuid']:raise ValueError('physicalGPU0 UUID changed')
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
    busy=any(line.split(',')[0].strip()==c['gpu_uuid'] for line in apps.splitlines())
    return {'uuid':selected[0][1],'free_mib':int(selected[0][2]),'compute_busy':busy,'ready':not busy and int(selected[0][2])>=30000}


def stage(c,name,args,env):
    begin=now()
    with (ROOT/'runs'/(name+'.log')).open('xb') as log:
        process=subprocess.Popen([c['python'],'-u','-B',*args],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT)
        put(ROOT/'runs'/(name+'.pid.json'),{'pid':process.pid,'started_at':begin,'args':args})
        try:code=process.wait(timeout=c['stage_timeout_seconds'])
        except subprocess.TimeoutExpired:
            process.terminate()
            try:process.wait(timeout=30)
            except subprocess.TimeoutExpired:process.kill();process.wait()
            code=124
    put(ROOT/'runs'/(name+'.exit.json'),{'exit_code':code,'pid':process.pid,'started_at':begin,'ended_at':now()})
    if code:raise RuntimeError(name+' failed exit '+str(code))
    event('stage_completed',stage=name)


def run():
    import fcntl
    m=verify();c=m['config'];deadline=time.monotonic()+c['wait_hours']*3600
    event('WAITING_TRAINING',parent=c['training_root']);rc=1;lock=None
    try:
        while not cpu_proof(c):
            if time.monotonic()>=deadline:raise TimeoutError('training wait deadline reached; no evaluation started')
            time.sleep(c['poll_seconds'])
        event('TRAINING_ACCEPTED_WAITING_GPU0')
        lock=open(c['shared_lock'],'a+')
        while True:
            acquired=False
            try:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);acquired=True
                status=resource(c)
                if status['ready']:
                    put(ROOT/'runs/resource-gate.json',status);break
            except BlockingIOError:pass
            if acquired:fcntl.flock(lock,fcntl.LOCK_UN)
            if time.monotonic()>=deadline:raise TimeoutError('GPU0 unavailable until wait deadline; no preemption')
            time.sleep(c['poll_seconds'])
        verify()
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',PYTHONPATH=c['deps'],PYTHONDONTWRITEBYTECODE='1',
            HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
            PYTHONHASHSEED='0',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8')
        for cohort in ('smoke','val80'):
            event('STARTING_'+cohort.upper())
            stage(c,cohort,['evaluate.py','--cohort',cohort],env)
            stage(c,cohort+'-audit',['replay.py','--run','runs/'+cohort,'--output','reviews/'+cohort+'-audit.json','--tokenizer'],dict(env,CUDA_VISIBLE_DEVICES=''))
        put(ROOT/'runs/completion.json',{'paired_val80_sources':80,'formal_traces':160,'smoke_traces':4,
            'full_replay_accepted':True,'evaluation_manifest_sha256':sha(ROOT/'manifest.json'),
            'training_anchor':read(ROOT/'runs/training-anchor.json'),'gold_never_uploaded':True,'ended_at':now()})
        rc=0
    except Exception as exc:
        traceback.print_exc();put(ROOT/'runs/failure.json',{'type':type(exc).__name__,'message':str(exc),'time':now()})
    finally:
        if lock:lock.close()
    put(ROOT/'runs/exit.json',{'exit_code':rc,'pid':os.getpid(),'ended_at':now()})
    result=subprocess.run([c['python'],'-B',str(ROOT/'archive.py')],cwd=ROOT,stdin=subprocess.DEVNULL)
    put(ROOT/'runs/archive.exit.json',{'exit_code':result.returncode,'ended_at':now()})
    return rc or result.returncode


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('launch','run'));a=p.parse_args()
    if a.mode=='run':raise SystemExit(run())
    m=verify();put(ROOT/'runs/launch-claim.json',{'manifest_sha256':sha(ROOT/'manifest.json'),'time':now()})
    with (ROOT/'runs/watch.log').open('xb') as log:
        worker=subprocess.Popen([m['config']['python'],'-u','-B',str(ROOT/'watch.py'),'run'],cwd=ROOT,
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    receipt={'pid':worker.pid,'manifest_sha256':sha(ROOT/'manifest.json'),'started_at':now(),'detached':True}
    put(ROOT/'runs/worker.json',receipt);print(__import__('json').dumps(receipt))


if __name__=='__main__':main()
