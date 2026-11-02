"""Unique detached CPU supervisor: accept old80, lock GPU0, run/replay new342 once."""
import argparse
import datetime
import os
from pathlib import Path
import subprocess
import time
import traceback
from io_utils import ROOT, read, put, sha, verify
from full_validation import copy_exclusive
from parent_acceptance import (parent_evidence, actual_training, model_identity, process_live,
                               TRAINING_FILES, execution_config_identity)


def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()


def live(pid, needle): return process_live(pid, needle)


def event(kind, **value):
    folder = ROOT/'runs/events'; folder.mkdir(parents=True, exist_ok=True)
    put(folder/('%05d.json' % (len(list(folder.glob('*.json')))+1)), {'kind':kind, 'time':now(), **value})
    print(kind, now(), flush=True)


def accept_parent(c):
    root = Path(c['parent_root']); out = root/'exports/final-01'
    execution_config_identity(read(root/'manifest.json')['config'], c)
    proof = parent_evidence(root, out/'student-val80-evidence.tar.gz', out/'members.json',
                            read(out/'archive.json'), read(root/'runs/archive.exit.json'), check_writers=True)
    if proof != read(ROOT/'data/reuse-parent.json'): raise ValueError('actual parent differs from frozen reuse proof')
    actual_training(c, proof['training_anchor'])
    base = model_identity(c)
    # Preserve original proof and anchor bytes without invoking or regenerating any training validator.
    copy_exclusive(root/'runs/training-anchor.json', ROOT/'runs/training-anchor.json')
    for name in TRAINING_FILES: copy_exclusive(root/'evidence/training'/name, ROOT/'evidence/training'/name)
    put(ROOT/'runs/parent-acceptance.json', {'parent_manifest_sha256':proof['parent_manifest_sha256'],
        'parent_archive_sha256':proof['parent_archive_sha256'], 'parent_members_sha256':proof['parent_members_sha256'],
        'parent_anchor_sha256':proof['parent_anchor_sha256'], 'old80_trace_count':160,
        'all_parent_members_rechecked':True, 'parent_writers_stopped':True, 'actual_A_unchanged':True,
        'base_identity':base, 'training_proof_reused_without_generation':True, 'accepted_at':now()})


def resource(c):
    rows = subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'], text=True)
    selected = [list(map(str.strip,line.split(','))) for line in rows.splitlines() if line.split(',')[0].strip()==c['gpu_index']]
    if len(selected)!=1 or selected[0][1]!=c['gpu_uuid']: raise ValueError('physical GPU0 UUID changed')
    apps = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'], text=True)
    busy = any(line.split(',')[0].strip()==c['gpu_uuid'] for line in apps.splitlines())
    return {'uuid':selected[0][1], 'free_mib':int(selected[0][2]), 'compute_busy':busy,
            'ready':not busy and int(selected[0][2])>=30000}


def stage(c, name, args, env):
    begin = now()
    with (ROOT/'runs'/(name+'.log')).open('xb') as log:
        process = subprocess.Popen([c['python'],'-u','-B',*args], cwd=ROOT, env=env,
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        put(ROOT/'runs'/(name+'.pid.json'), {'pid':process.pid, 'started_at':begin, 'args':args})
        try: code = process.wait(timeout=c['stage_timeout_seconds'])
        except subprocess.TimeoutExpired:
            process.terminate()
            try: process.wait(timeout=30)
            except subprocess.TimeoutExpired: process.kill(); process.wait()
            code = 124
    put(ROOT/'runs'/(name+'.exit.json'), {'exit_code':code, 'pid':process.pid, 'started_at':begin, 'ended_at':now()})
    if code: raise RuntimeError(name+' failed exit '+str(code))
    event('stage_completed', stage=name)


def run():
    import fcntl
    m = verify(); c = m['config']; deadline = time.monotonic()+c['wait_hours']*3600
    rc = 1; lock = None
    try:
        claim = read(ROOT/'runs/launch-claim.json')
        if claim['manifest_sha256'] != sha(ROOT/'manifest.json'): raise ValueError('launch manifest differs')
        # Also excludes manual duplicate `run` invocations after the unique launch.
        put(ROOT/'runs/run-claim.json', {'pid':os.getpid(), 'manifest_sha256':claim['manifest_sha256'], 'time':now()})
        event('ACCEPTING_COMPLETED_PARENT80'); accept_parent(c)
        event('PARENT80_ACCEPTED_WAITING_GPU0'); lock = open(c['shared_lock'], 'a+')
        while True:
            acquired = False
            try:
                fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB); acquired = True
                status = resource(c)
                if status['ready']: put(ROOT/'runs/resource-gate.json', status); break
            except BlockingIOError: pass
            if acquired: fcntl.flock(lock, fcntl.LOCK_UN)
            if time.monotonic() >= deadline: raise TimeoutError('GPU0 unavailable at 24h deadline; no run started')
            time.sleep(c['poll_seconds'])
        verify(); actual_training(c, read(ROOT/'runs/training-anchor.json'))
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', PYTHONPATH=c['deps'], PYTHONDONTWRITEBYTECODE='1',
            HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
            PYTHONHASHSEED='0', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8')
        event('STARTING_VAL342')
        stage(c, 'val342', ['evaluate.py','--cohort','val342'], env)
        stage(c, 'val342-audit', ['replay.py','--run','runs/val342','--output','reviews/val342-audit.json','--tokenizer'],
              dict(env, CUDA_VISIBLE_DEVICES=''))
        audit = read(ROOT/'reviews/val342-audit.json'); summary = read(ROOT/'runs/val342/summary.json')
        if (audit['sources']!=342 or audit['traces']!=684 or not audit['actual_route_and_tokens_verified']
                or summary['paired_sources']!=342 or summary['status']!='completed'):
            raise ValueError('supplement342 full coverage/replay differs')
        put(ROOT/'runs/completion.json', {'supplementary_sources':342, 'supplementary_traces':684,
            'reused_formal_sources':80, 'reused_formal_traces':160, 'expected_full_sources':422,
            'expected_full_traces':844, 'full_replay_accepted':True, 'supplementary_complete':True,
            'evaluation_complete':False, 'full422_pending_local_merge':True,
            'evaluation_manifest_sha256':sha(ROOT/'manifest.json'),
            'parent_archive_sha256':c['parent_archive_sha256'], 'training_anchor':read(ROOT/'runs/training-anchor.json'),
            'gold_never_uploaded':True, 'ended_at':now()})
        rc = 0
    except FileExistsError:
        # A duplicate runner cannot write failure/exit/archive on behalf of the legitimate writer.
        if (ROOT/'runs/run-claim.json').exists() and read(ROOT/'runs/run-claim.json')['pid'] != os.getpid():
            raise
        traceback.print_exc(); put(ROOT/'runs/failure.json', {'type':'FileExistsError', 'message':'exclusive output already exists', 'time':now(), 'expected_sources':422})
    except Exception as exc:
        traceback.print_exc(); put(ROOT/'runs/failure.json', {'type':type(exc).__name__, 'message':str(exc), 'time':now(), 'expected_sources':422})
    finally:
        if lock: lock.close()
    put(ROOT/'runs/exit.json', {'exit_code':rc, 'pid':os.getpid(), 'ended_at':now()})
    result = subprocess.run([c['python'],'-B',str(ROOT/'archive.py')], cwd=ROOT, stdin=subprocess.DEVNULL)
    put(ROOT/'runs/archive.exit.json', {'exit_code':result.returncode, 'ended_at':now()})
    return rc or result.returncode


def main():
    p = argparse.ArgumentParser(); p.add_argument('mode', choices=('launch','run')); a = p.parse_args()
    if a.mode == 'run': raise SystemExit(run())
    m = verify()
    # Exclusive claim happens before spawning. There is no recovery/restart path.
    put(ROOT/'runs/launch-claim.json', {'manifest_sha256':sha(ROOT/'manifest.json'), 'time':now()})
    with (ROOT/'runs/watch.log').open('xb') as log:
        worker = subprocess.Popen([m['config']['python'],'-u','-B',str(ROOT/'watch.py'),'run'], cwd=ROOT,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    receipt = {'pid':worker.pid, 'manifest_sha256':sha(ROOT/'manifest.json'), 'started_at':now(), 'detached':True}
    put(ROOT/'runs/worker.json', receipt); print(__import__('json').dumps(receipt))


if __name__ == '__main__': main()
