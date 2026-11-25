"""One prospective20-state collection. No retries, training, keys on disk, or old writes."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import threading
import time
import urllib.request
import urllib.error

ROOT = Path(__file__).resolve().parent
AUDIT = ROOT.parent
REPO = AUDIT.parent
R = AUDIT / 'finance_program_student_grpo_20261017_r3'
SFT = AUDIT / 'finance_program_student_sft_20261016_r1/reviews/final-01/extract/data/planner200.jsonl'
Q = AUDIT / 'finance_program_conditional_sft_audit_20261018_r2/reviews/conditional-attempt-review-01.json'
API = AUDIT / 'finance_program_teacher_balanced_20261016_r9/lib/teacher_api.py'
OLD_CONFIG = AUDIT / 'finance_program_teacher_balanced_20261016_r9/config.json'
FIELDS = {'database_name','database_description','table_name','table_description','field_name','field_description','field_type','value'}
GENERATION = {'input_tokens','output_tokens','error','truncation_at_budget'}


def raw_json(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as f:
        f.write(raw_json(value)); f.flush(); os.fsync(f.fileno())


def read(path):
    return json.loads(Path(path).read_bytes())


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def program_shape(text):
    def unique(pairs):
        result = {}
        for k,v in pairs:
            if k in result: raise ValueError('duplicate key')
            result[k] = v
        return result
    out = dict(json_valid=False, program_shape=False, complete_intent=False, path=[], error=None)
    try:
        v = json.loads(text, object_pairs_hook=unique)
        out['json_valid'] = True
        if not isinstance(v,dict) or set(v) != {'program'} or not isinstance(v['program'],list) or not v['program']:
            raise ValueError('exact nonempty Program required')
        for s in v['program']:
            if not isinstance(s,dict) or set(s) != {'tool','arguments'} or not isinstance(s['tool'],str) or not isinstance(s['arguments'],dict):
                raise ValueError('invalid step')
            out['path'].append(s['tool'])
        out['program_shape'] = True
        out['complete_intent'] = out['path'][-1] == 'FINALIZE_CLASSIFICATION'
    except (ValueError,TypeError) as e:
        out['error'] = str(e)
    return out


def generation(md):
    if 'generation' in md:
        return deepcopy(md['generation'])
    has_flat = any(k in md for k in GENERATION - {'error'})
    return {k:deepcopy(v) for k,v in md.items() if k in GENERATION and (k != 'error' or has_flat)}


def prefix_calls(trace, target_attempt):
    prefix = []
    number = 0
    for a in trace['planner_attempts'][:target_attempt-1]:
        prefix.append(dict(messages=a['messages'], response=a.get('raw_response'),
                           error=a.get('error'), generation={}))
        if a['status'] != 'accepted': continue
        number += 1
        p = next(p for p in trace['programs'] if p['program_index'] == number)
        for step in p['steps']:
            md = step.get('metadata') or {}
            calls = md.get('attempts', [md] if 'messages' in md else [])
            for c in calls:
                prefix.append(dict(messages=c['messages'], response=c.get('raw_response'),
                                   error=c.get('error') or step.get('summary'), generation=generation(c)))
    return deepcopy(prefix)


def validate_requests(items):
    if len(items) != 20: raise ValueError('exact20 request denominator required')
    if len({x['id'] for x in items}) != 20: raise ValueError('duplicate request')
    if Counter(x['kind'] for x in items) != {'initial':12, 'recovery':6, 'continuation':2}:
        raise ValueError('20-state mix changed')


class SnapshotReady(BaseException):
    pass


class Router:
    def __init__(self, item, response=None, skill=None):
        self.item,self.response,self.skill = item,response,skill
        self.pos = self.new_planner_calls = self.new_skill_calls = 0
        self.last_generation_metadata = {}
        self.target_exact = False
        self.calls = []
        self.failed = False

    def __call__(self, messages):
        self.last_generation_metadata = {}
        if self.failed: raise RuntimeError('historical mismatch; stopped')
        if self.pos < len(self.item['prefix']):
            call = self.item['prefix'][self.pos]
            if raw_json(messages) != raw_json(call['messages']):
                self.failed = True
                raise RuntimeError('historical prefix exact request mismatch')
            self.pos += 1
            self.last_generation_metadata = deepcopy(call['generation'])
            if call['response'] is None: raise RuntimeError(call['error'])
            return call['response']
        if not self.target_exact:
            if raw_json(messages) != raw_json(self.item['messages']):
                self.failed = True
                raise RuntimeError('target exact request mismatch')
            self.target_exact = True
            if self.response is None: raise SnapshotReady()
            self.new_planner_calls += 1
            return self.response
        if messages[0]['content'].startswith('You are the Global Program Planner'):
            raise RuntimeError('one teacher Program only; no completion repair')
        self.new_skill_calls += 1
        try:
            text = self.skill(messages)
            self.last_generation_metadata = deepcopy(getattr(self.skill,'last_generation_metadata',{}))
            self.calls.append(dict(messages=deepcopy(messages),response=text,generation=deepcopy(self.last_generation_metadata)))
            return text
        except Exception as exc:
            self.calls.append(dict(messages=deepcopy(messages),response=None,error=type(exc).__name__,generation=deepcopy(self.last_generation_metadata)))
            raise


def controller(router, study, runtime=R/'runtime'):
    sys.path.insert(0,str(runtime))
    from program_loop.controller import build_program_controller
    return build_program_controller(router, study['registry'], study['rows'], study['budget'],skill_pool='F66')


def prepare():
    if (ROOT/'inputs').exists(): raise FileExistsError('inputs already selected; reuse exact lock')
    study_raw=(R/'data/study.json').read_bytes(); study=json.loads(study_raw)
    assert sha(SFT.read_bytes()) == 'f93ffa2dd9ff8aae06c3872a1e4c0a6c532b7e733c485912ec0740e052cd0afa'
    assert sha(Q.read_bytes()) == 'd4393e4b253575a6348538560f4369946aa2a38dc55838e337291df21a20e608'
    sft=[json.loads(line) for line in SFT.read_text(encoding='utf-8').splitlines()]
    allowed={v['source_id'] for v in sft}; byid={v['id']:v for v in study['rows']}
    assert len(sft)==200 and allowed <= set(byid)
    items=[]; groups=set()
    for row in sft:
        sid=row['source_id']; source=byid[sid]
        group=(source['domain'],source['metadata'].get('table_name') or sid)
        if group in groups: continue
        groups.add(group)
        # Use actual student initial prompt, not an old teacher-specific prompt.
        class InitialCapture(BaseException): pass
        captured=[]
        def capture(msg): captured.append(msg); raise InitialCapture()
        try: controller(capture,study).run(dict(source_id=sid,domain=source['domain'],field_name=source['metadata']['field_name']))
        except InitialCapture: pass
        items.append(dict(id=f'initial-{len(items)+1:02d}',kind='initial',source_id=sid,
            initial=dict(source_id=sid,domain=source['domain'],field_name=source['metadata']['field_name']),
            messages=captured[0],prefix=[],provenance={'sft_row':sft.index(row),'original_teacher_trace_sha256':row['trace_sha256']}))
        if len(items)==12: break
    cases=read(Q)['cases']
    for kind,status,count in [('recovery','failed',6),('continuation','completed',2)]:
        seen=set(); selected=0
        for c in cases:
            if c['source_id'] in seen: continue
            path=Path(c['trace_path']); raw=path.read_bytes()
            assert sha(raw)==c['trace_sha256'] and path.resolve().is_relative_to(R.resolve())
            t=json.loads(raw)
            for a in t['planner_attempts'][1:]:
                payload=json.loads(a['messages'][-1]['content'])
                fb=payload.get('feedback') or {}
                if fb.get('status') != status: continue
                sid=c['source_id']; assert sid in allowed
                items.append(dict(id=f'{kind}-{selected+1:02d}',kind=kind,source_id=sid,
                    initial={**t['input'],'source_id':sid},messages=a['messages'],prefix=prefix_calls(t,a['attempt']),
                    provenance=dict(case=c['case'],candidate=c['candidate'],attempt=a['attempt'],
                        trace_path=path.relative_to(REPO).as_posix(),trace_sha256=sha(raw))))
                seen.add(sid);selected+=1;break
            if selected==count: break
        if selected!=count: raise ValueError('not enough genuine states; do not fabricate')
    for item in items:
        item['source']={k:deepcopy(v) for k,v in byid[item['source_id']].items() if k in {'id','domain','metadata'}}
        item['source']['metadata']={k:v for k,v in item['source']['metadata'].items() if k in FIELDS}
        item['messages_sha256']=sha(raw_json(item['messages']))
        item['group']=[item['source']['domain'],item['source']['metadata'].get('table_name') or item['source_id']]
        probe=Router(item)
        try: controller(probe,study).run(item['initial'])
        except SnapshotReady: pass
        if not probe.target_exact or probe.pos!=len(item['prefix']):
            raise ValueError('actualCPU prefix/target mismatch: '+item['id'])
    random.Random(20261019).shuffle(items)
    validate_requests(items)
    for n,item in enumerate(items,1): item['order']=n
    save(ROOT/'inputs/requests.json',items)
    save(ROOT/'inputs/study.json',study)
    config=read(OLD_CONFIG)
    config={k:config[k] for k in ('model','endpoint','thinking_budget','max_tokens','timeout_seconds')}
    config.update(request_cap=20,total_token_cap=1000000,concurrency=2,temperature=0,transport_retries=0)
    save(ROOT/'config.json',config)
    dependencies=[SFT,Q,R/'data/study.json',API,OLD_CONFIG,
                  AUDIT/'finance_program_teacher_expand_20261015_r1/credentials.py']
    dependencies += list((R/'runtime').rglob('*.py'))
    dependencies += list((R/'runtime').rglob('*.json'))
    dependencies += list((R/'runtime').rglob('*.md'))
    dependencies += [REPO/i['provenance']['trace_path'] for i in items if 'trace_path' in i['provenance']]
    own=[ROOT/'collect20.py',ROOT/'config.json',ROOT/'inputs/requests.json',ROOT/'inputs/study.json',
         REPO/'docs/superpowers/specs/2026-10-19-bailian-state20-design.md',REPO/'docs/superpowers/plans/2026-10-19-bailian-state20.md']
    files=[dict(path=p.resolve().as_posix(),size=p.stat().st_size,sha256=sha(p.read_bytes())) for p in dict.fromkeys(dependencies+own)]
    save(ROOT/'protocol-lock.json',dict(schema='state20-lock-v1',files=files,states=20,cpu_prefixes_exact=20,
        sources=len({i['source_id'] for i in items}),groups=len({tuple(i['group']) for i in items}),mix=dict(Counter(i['kind'] for i in items))))
    print(json.dumps(dict(states=20,cpu_prefixes_exact=20,mix=dict(Counter(i['kind'] for i in items)),sources=len({i['source_id'] for i in items}))))


def verify():
    for b in read(ROOT/'protocol-lock.json')['files']:
        p=Path(b['path']); raw=p.read_bytes()
        if len(raw)!=b['size'] or sha(raw)!=b['sha256']: raise ValueError('locked dependency changed: '+p.name)
    items=read(ROOT/'inputs/requests.json');validate_requests(items)
    return items


def collect():
    items=verify(); config=read(ROOT/'config.json')
    api=load('original_sse20',API)
    credentials=load('authorized_teacher_credential20',AUDIT/'finance_program_teacher_expand_20261015_r1/credentials.py')
    (ROOT/'collection').mkdir(exist_ok=False)
    save(ROOT/'collection/start.json',dict(pid=os.getpid(),time_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),lock_sha256=sha((ROOT/'protocol-lock.json').read_bytes()),credential_recorded=False))
    key=credentials.read_key()
    guard=threading.Lock(); budget={'claimed':0,'known_tokens':0,'halt':False}
    def one(item):
        folder=ROOT/'collection'/item['id'];folder.mkdir(exist_ok=False)
        with guard:
            if budget['halt'] or budget['claimed']>=20 or budget['known_tokens']>=1000000:
                result={'complete':False,'error_type':'GlobalBudgetHalt'}
                save(folder/'result.json',result);return result
            budget['claimed']+=1
        messages=deepcopy(item['messages'])
        messages[0]['content']+='\nTransport output contract: return exactly one raw JSON object; no Markdown or prose outside JSON. Preserve the original planning goal and all budgets.'
        body=dict(model=config['model'],messages=messages,stream=True,stream_options={'include_usage':True},
                  temperature=0,thinking_budget=config['thinking_budget'],max_tokens=config['max_tokens'],response_format={'type':'json_object'})
        save(folder/'request.json',dict(body=body,endpoint=config['endpoint'],source_messages_sha256=item['messages_sha256'],credential_recorded=False))
        result={};started=time.monotonic()
        try:
            request=urllib.request.Request(config['endpoint']+'/chat/completions',data=raw_json(body),
                headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'},method='POST')
            with urllib.request.urlopen(request,timeout=config['timeout_seconds']) as response:
                with (folder/'response.sse').open('xb') as stream:
                    def lines():
                        for raw in response:
                            stream.write(raw);stream.flush()
                            if time.monotonic()-started>config['timeout_seconds']: raise TimeoutError()
                            yield raw
                    result=api.parse_sse(lines(),config['model'],state=result)
                    result['http_status']=response.status
        except api.CompletionError as exc:
            result=exc.result;result['error_type']='IncompleteCompletion'
        except urllib.error.HTTPError as exc:
            result.update(complete=False,error_type='HTTPError',http_status=exc.code)
            with guard: budget['halt']=True
        except Exception as exc:
            result.update(complete=False,error_type=type(exc).__name__)
        result.update(seconds=time.monotonic()-started,source_id=item['source_id'],state_id=item['id'],
                      quality=program_shape(result.get('content','')),credential_recorded=False)
        with guard: budget['known_tokens']+=(result.get('usage') or {}).get('total_tokens',0)
        save(folder/'result.json',result)
        print(json.dumps(dict(state=item['id'],complete=result.get('complete',False),seconds=round(result['seconds'],2))),flush=True)
        return result
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(one,i) for i in items]
        results=[f.result() for f in as_completed(futures)]
    del key
    save(ROOT/'collection/summary.json',dict(expected_states=20,records=len(results),complete=sum(bool(r.get('complete')) for r in results),budget=budget,
        total_seconds=sum(r.get('seconds',0) for r in results),wall_completed_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()))
    print(json.dumps(read(ROOT/'collection/summary.json')))


if __name__=='__main__':
    {'prepare':prepare,'collect':collect}[sys.argv[1]]()
