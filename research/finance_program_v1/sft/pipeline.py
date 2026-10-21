"""One detached, mutually exclusive trial -> fresh formal training pipeline."""
# 持有 GPU 互斥锁依次启动 trial 和 formal；正式训练须通过试跑证据门禁，每阶段独立记录退出状态。
import argparse
import datetime
import os
from pathlib import Path
import subprocess
import sys
import traceback

from runtime_io import digest, put, read, verify_manifest
from training_contract import validate_gpu_snapshot, validate_trial


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def claim_launch(root, signature):
    put(Path(root) / 'runs/launch-claim.json', {'manifest_sha256': signature, 'claimed_at': now()})


def formal_gate(trial_exit, proof, config):
    if trial_exit.get('exit_code') != 0 or not isinstance(proof, dict):
        raise ValueError('actual successful trial exit and proof required')
    validate_trial(proof, config)


def gpu_snapshot(expected):
    values = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.free',
        '--format=csv,noheader,nounits'], text=True)
    observed, free = {}, {}
    for line in values.strip().splitlines():
        idx, uuid, memory = [x.strip() for x in line.split(',')]
        if idx in expected:
            observed[idx], free[idx] = uuid, int(memory)
    busy_text = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
        '--format=csv,noheader,nounits'], text=True)
    busy = [line.split(',')[0].strip() for line in busy_text.splitlines() if line.strip()]
    validate_gpu_snapshot(expected, observed, free, busy)
    return {'observed': observed, 'free_mib': free, 'busy_gpu_uuids': busy, 'checked_at': now()}


def run(root):
    import fcntl
    m = verify_manifest(root)
    c = m['config']
    handles = []
    rc = 1
    try:
        for idx in sorted(c['gpu_uuids']):
            f = open(c['lock_dir'] + '/gpu' + idx + '-experiment.lock', 'a+')
            handles.append(f)
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        put(root / 'runs/resource-gate.json', gpu_snapshot(c['gpu_uuids']))
        if subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], text=True).strip() == '':
            raise RuntimeError('GPU driver query failed')
        from shutil import disk_usage
        if disk_usage(root).free < 20 * 1024 ** 3:
            raise RuntimeError('less than 20 GiB disk available')
        identity = read(root / 'model-identity.json')
        for item in identity['files']:
            if digest(Path(c['model']) / item['path']) != item['sha256']:
                raise ValueError('base model identity changed: ' + item['path'])
        put(root / 'runs/base-identity-proof.json', {'all_files_verified': True,
            'revision': identity['revision'], 'identity_sha256': digest(root / 'model-identity.json'), 'time': now()})
        env = dict(os.environ)
        env.update(CUDA_VISIBLE_DEVICES=','.join(sorted(c['gpu_uuids'])), PYTHONUTF8='1',
            PYTHONDONTWRITEBYTECODE='1', TOKENIZERS_PARALLELISM='false',
            PYTHONPATH=c['deps'], OMP_NUM_THREADS='8', MKL_NUM_THREADS='8',
            HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        # 每阶段启动独立训练进程，formal 重新加载基座，不继承 trial 的权重或优化器。
        for stage in ('trial', 'formal'):
            if stage == 'formal':
                formal_gate(read(root / 'runs/trial.exit.json'), read(root / 'trial/proof.json'), c)
            started = now()
            with (root / 'runs' / (stage + '.log')).open('xb') as log:
                worker = subprocess.Popen([c['python'], '-u', '-B', str(root / 'train_student.py'), '--mode', stage],
                    cwd=root, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                put(root / 'runs' / (stage + '.pid.json'), {'pid': worker.pid, 'stage': stage, 'started_at': started})
                code = worker.wait()
            put(root / 'runs' / (stage + '.exit.json'), {'exit_code': code, 'pid': worker.pid,
                'started_at': started, 'ended_at': now()})
            if code:
                raise RuntimeError(stage + ' exited with code ' + str(code))
        complete = read(root / 'formal/completion.json')
        if complete['global_step'] != 13 or complete['sources'] != 200 or complete['epoch'] != 1.0:
            raise RuntimeError('formal training incomplete')
        rc = 0
    except Exception as exc:
        traceback.print_exc()
        put(root / 'runs/failure.json', {'type': type(exc).__name__, 'message': str(exc), 'time': now()})
    finally:
        for f in reversed(handles):
            f.close()
    put(root / 'runs/pipeline.exit.json', {'exit_code': rc, 'ended_at': now(), 'pid': os.getpid()})
    # Stage writers have exited; archive regardless of success, without restarting training.
    archive = subprocess.run([c['python'], '-B', str(root / 'archive_run.py')], cwd=root,
        stdin=subprocess.DEVNULL, check=False)
    put(root / 'runs/archive.exit.json', {'exit_code': archive.returncode, 'ended_at': now()})
    return rc if rc else archive.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('launch', 'run'))
    a = parser.parse_args()
    root = Path(__file__).resolve().parent
    if a.mode == 'run':
        raise SystemExit(run(root))
    m = verify_manifest(root)
    claim_launch(root, digest(root / 'manifest.json'))
    with (root / 'runs/pipeline.log').open('xb') as log:
        worker = subprocess.Popen([m['config']['python'], '-u', '-B', str(root / 'pipeline.py'), 'run'],
            cwd=root, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    receipt = {'pid': worker.pid, 'manifest_sha256': digest(root / 'manifest.json'), 'started_at': now(),
        'detached': True, 'no_automatic_restart': True}
    put(root / 'runs/worker.json', receipt)
    print(__import__('json').dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
