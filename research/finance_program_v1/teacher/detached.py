"""CPU-only detached continuation. Raw trajectories stay pending human review."""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parent
# Controlled historical snapshots own runtime/; the public layout shares the
# exact F66 snapshot already shipped under evaluation/runtime, not a new copy.
RUNTIME = ROOT/'runtime'
if not RUNTIME.is_dir():
    RUNTIME = ROOT.parent/'evaluation/runtime'
sys.path[:0] = [str(ROOT/'lib'), str(RUNTIME)]
from campaign_budget import PersistentBudget, RunBudget
from teacher_api import TeacherBackend, save
from retry_backend import RetryBackend


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def select_ids(rows, used, count):
    ids=[r['id'] for r in rows]
    if len(ids)!=len(set(ids)):raise ValueError('duplicate source')
    assert 0 < count <= 200
    return [i for i in ids if i not in used][:count]


def verify_files(root, manifest):
    for item in manifest['files']:
        path=(root/item['path']).resolve()
        assert path.is_relative_to(root.resolve()) and sha(path)==item['sha256'], item['path']


def verify():
    manifest=read(ROOT/'manifest.json');verify_files(ROOT,manifest)
    from runtime_identity import verify as verify_runtime
    assert verify_runtime()['core_signature_sha256']==manifest['runtime_signature']
    history=(ROOT/'ledger/events.jsonl').read_bytes()
    assert hashlib.sha256(history[:manifest['ledger_prefix_bytes']]).hexdigest()==manifest['ledger_prefix_sha256']
    study=read(ROOT/'study.json')
    assert all(set(r)=={'id','domain','metadata'} for r in study['rows'])
    assert all(set(r['metadata']) <= {'database_name','database_description','table_name','table_description','field_name','field_description','field_type','value'} for r in study['rows'])
    assert not set(manifest['source_ids']) & set(manifest['used_ids'])
    assert len(set(manifest['source_ids']))==len(manifest['source_ids'])
    return manifest


def seal(root):
    archive=root/'evidence.tar.gz';receipt=root/'archive-receipt.json'
    if archive.exists() or receipt.exists():raise FileExistsError('archive exists')
    paths=sorted(p for p in root.rglob('*') if p.is_file())
    entries=[{'path':p.relative_to(root).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size} for p in paths]
    with archive.open('xb') as stream:
        with tarfile.open(fileobj=stream, mode='w:gz') as tar:
            for p,e in zip(paths,entries):tar.add(p,arcname=e['path'],recursive=False)
    with tarfile.open(archive) as tar:
        members=tar.getmembers()
        assert len(members)==len(entries)
        for member,e in zip(members,entries):
            assert member.isfile() and member.name==e['path']
            assert hashlib.sha256(tar.extractfile(member).read()).hexdigest()==e['sha256']
    result={'sha256':sha(archive),'members':len(entries),'files':entries,'all_members_sha_verified':True}
    save(receipt,result)
    return result


def replay():
    verify()
    from program_loop import build_program_controller
    from pilot_audit import read_records, replay_retry_one
    study=read(ROOT/'study.json');rows={r['id']:r for r in study['rows']}
    c=build_program_controller(lambda m:'',study['registry'],study['rows'],study['budget'],skill_pool='F66')
    allowed=[s['name'] for s in c.library];assert len(set(allowed))==66
    proofs=[]
    for folder in sorted((ROOT/'replay-fixtures').iterdir()):
        records=read_records(folder/'calls')
        logical=[read(p) for p in sorted((folder/'logical').glob('logical-*.json'))]
        proofs.append(replay_retry_one(study,rows[folder.name],read(folder/'trace.json'),records,logical,allowed))
    assert proofs and all(p['full_trace_exact'] and p['all_calls_consumed'] for p in proofs)
    result={'sources':len(proofs),'full_trace_exact':True,'http_calls':sum(p['api_attempts'] for p in proofs),'paid_calls':0,'runtime_signature':read(ROOT/'manifest.json')['runtime_signature']}
    return result


def run_batch(number, ids, key):
    from program_loop import build_program_controller
    from pilot_audit import replay_retry_one
    study=read(ROOT/'study.json');config=read(ROOT/'config.json');rows={r['id']:r for r in study['rows']}
    root=ROOT/'batches'/('batch-%03d'%number);root.mkdir(parents=True,exist_ok=False)
    save(root/'manifest.json',{'ids':ids,'started_utc':now(),'manifest_sha256':sha(ROOT/'manifest.json'),'gold_in_input':False,'student_training':False})
    c=build_program_controller(lambda m:'',study['registry'],study['rows'],study['budget'],skill_pool='F66')
    allowed=[s['name'] for s in c.library];assert len(set(allowed))==66
    budget=RunBudget(ROOT/'ledger',requests=config['request_cap'],tokens=config['total_token_cap'],failures=config['failed_attempt_cap'])
    def source(i):
        folder=root/'sources'/i;folder.mkdir(parents=True,exist_ok=False)
        print('source start',i,flush=True)
        backend=RetryBackend(TeacherBackend(key,config,folder/'calls',allowed),config['model'],folder/'logical',budget)
        row=rows[i];controller=build_program_controller(backend,study['registry'],study['rows'],study['budget'],skill_pool='F66')
        trace=controller.run({'source_id':i,'domain':row['domain'],'field_name':row['metadata']['field_name']})
        trace.write_json(folder/'trace.json')
        proof=replay_retry_one(study,row,trace.to_dict(),backend.calls,backend.logical,allowed)
        save(folder/'replay.json',proof)
        entry={'id':i,'status':'pending_review','terminated':trace.terminated,'stop_reason':trace.stop_reason,'http_attempts':len(backend.calls),'logical_calls':len(backend.logical),'accepted':None}
        save(folder/'receipt.json',entry);print('source done',i,trace.stop_reason,flush=True)
        return entry
    started=time.monotonic();entries={};queue=iter(ids);failed=False
    with ThreadPoolExecutor(max_workers=config['concurrency']) as pool:
        active={}
        def fill():
            while len(active)<config['concurrency'] and not budget.snapshot()['stopped']:
                i=next(queue,None)
                if i is None:break
                active[pool.submit(source,i)]=i
        fill()
        while active:
            done,_=wait(active,return_when=FIRST_COMPLETED)
            for future in done:
                i=active.pop(future)
                try:entries[i]=future.result()
                except Exception as exc:
                    failed=True;budget.stop()
                    entries[i]={'id':i,'status':'engineering_failed','error_type':type(exc).__name__,'accepted':None}
            save(root/'progress'/('%04d.json'%len(entries)),{'samples':list(entries.values()),'budget':budget.snapshot()})
            fill()
    for i in ids:
        if i not in entries:entries[i]={'id':i,'status':'not_started_safety_stop','accepted':None}
    result={'status':'engineering_failed' if failed else 'pending_review','expected_ids':ids,'samples':[entries[i] for i in ids],'seconds':time.monotonic()-started,'budget':budget.snapshot(),'gold_in_input':False,'student_training':False,'automatic_semantic_admission':False}
    save(root/'summary.json',result);save(root/'exit.json',{'code':2 if failed else 0,'finished_utc':now()})
    shutil.copyfile(ROOT/'ledger/events.jsonl',root/'ledger-after.jsonl')
    receipt=seal(root)
    save(ROOT/'progress'/('batch-%03d.json'%number),{'batch':number,'sha256':receipt['sha256'],'status':result['status'],'budget':result['budget']})
    return result


def worker(key):
    manifest=verify();started=now();summary={'started_utc':started,'pid':os.getpid(),'accepted_before':manifest['accepted_before'],'automatic_semantic_admission':False,'student_training':False}
    save(ROOT/'worker.json',{**summary,'manifest_sha256':sha(ROOT/'manifest.json')})
    rc=0;reason='source_limit_reached'
    try:
        for offset in range(0,len(manifest['source_ids']),20):
            if shutil.disk_usage(ROOT).free<5_000_000_000:reason='disk_safety_stop';break
            if PersistentBudget(ROOT/'ledger').snapshot()['stopped']:reason='campaign_safety_stop';break
            result=run_batch(1+offset//20,manifest['source_ids'][offset:offset+20],key)
            if result['status']=='engineering_failed':rc=2;reason='engineering_failed';break
        if PersistentBudget(ROOT/'ledger').snapshot()['stopped']:reason='campaign_safety_stop'
    except Exception as exc:
        rc=2;reason='engineering_failed';save(ROOT/'failure.json',{'error_type':type(exc).__name__})
    save(ROOT/'completion.json',{**summary,'finished_utc':now(),'reason':reason,'budget':PersistentBudget(ROOT/'ledger').snapshot(),'pending_review':True})
    save(ROOT/'exit.json',{'code':rc,'reason':reason,'finished_utc':now()})
    return rc


def launch():
    import resource
    import signal
    import ctypes
    os.umask(0o077);resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    assert ctypes.CDLL(None).prctl(4,0,0,0,0)==0
    verify();assert read(ROOT/'replay-check.json')['full_trace_exact']
    assert read(ROOT/'tests-check.json')['exit_code']==0
    (ROOT/'launch-reservation').mkdir(exist_ok=False)
    key=sys.stdin.readline().strip()
    if not key.startswith('sk-'):raise ValueError('missing in-memory key')
    pid=os.fork()
    if pid:print(json.dumps({'detaching':True,'launcher_child':pid}),flush=True);return
    os.setsid();signal.signal(signal.SIGHUP,signal.SIG_IGN)
    if os.fork():os._exit(0)
    os.chdir(ROOT)
    stdin=os.open(os.devnull,os.O_RDONLY);log=os.open(ROOT/'run.log',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    os.dup2(stdin,0);os.dup2(log,1);os.dup2(log,2);os.close(stdin);os.close(log)
    os._exit(worker(key))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('verify','replay','launch'));args=parser.parse_args()
    if args.mode=='verify':print(json.dumps({'verified':bool(verify())}))
    elif args.mode=='replay':print(json.dumps(replay()))
    else:launch()
