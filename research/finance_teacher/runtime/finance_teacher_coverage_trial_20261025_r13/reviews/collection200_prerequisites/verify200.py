"""Independent raw/native/input/usage proof, without reading reference labels.

Current engineering prototype is pinned to the verified R12-derived 406 lookup.
It neither samples a model nor retries an existing HTTP slot.
"""
import copy,re,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parents[1]))
import native200
from r13_common import read,put,sha,binding,exact,ROOT,R9,R12
from r13_science import load_replay
def contract():
 from release_inputs import MANIFEST, INPUT
 return dict(native_factory=binding(HERE/'native200.py'),source_bundle=binding(INPUT),release_manifest=binding(MANIFEST),verifier=binding(__file__))
def verify(target,receipt):
 target,receipt=Path(target),Path(receipt)
 if receipt.exists():
  proof=read(receipt)
  if proof.get('runtime_contract')!=contract():raise ValueError('existing receipt is not this immutable 406 contract')
  for ref in list(proof['runtime_contract'].values())+proof['files']:exact(ref)
  return proof
 bundle=native200.bundle();rt=native200.runtime();camp=rt.previous.original.load('verify200_pinned_campaign',R9/'campaign.py')
 recovery=camp.load('verify200_pinned_recovery',R9/'recovery_policy.py');parser=camp.load('verify200_independent_SSE',camp.PARSER)
 wire=camp.load('verify200_pinned_wire_classification',ROOT.parent/'finance_teacher_program_coldstart_20261024_r5/reviews/wire_readiness.py')
 inputs=read(target/'program-inputs.json');config=read(target/'config.json');comp=read(target/'completion.json');os_exit=read(target/'OS-exit.json')
 if comp['actual_worker_exit']!=os_exit['actual_OS_exit']:raise ValueError('actual OS exit differs from worker completion')
 if config['model']!='qwen3.8-2.4t-a95b' or camp.canon(inputs['study'])!=camp.canon(bundle['study']):raise ValueError('frozen model or public study differs')
 testonly=comp.get('TestOnly') is True
 if not testonly and not re.fullmatch(r'https://[A-Za-z0-9-]+\.cn-beijing\.maas\.aliyuncs\.com/compatible-mode/v1',config['endpoint']):raise ValueError('non-TestOnly provider endpoint identity differs')
 allowed={c['source_id']:c for c in bundle['cases']};engine,scoped,_,_=rt.native_context();replay=load_replay();replay.CommonActor=rt.CommonActor;rows=[];calls=[];claimed=[];cost=0.;unknown=0;tokens=[0,0]
 seen=set()
 for case in inputs['cases']:
  if case['source_id'] in seen:raise ValueError('duplicate source in proof denominator')
  seen.add(case['source_id']);expected=allowed[case['source_id']]
  if case['case']!=expected['case'] or any(case[k]!=expected[k] for k in ('source_index','group','component','split')):raise ValueError('actual source initial or train-component differs')
  folder=target/'runs'/f"source-{case['source_index']:03d}";tracepath=folder/'trace.json';native_exact=False;trace={}
  if tracepath.exists():
   trace=read(tracepath);rep=replay.Replay(folder,config,trace['stop_reason']=='campaign_budget_exhausted')
   again=native200.build(engine,scoped,rep.stopped,bundle)(case['case']['item'],inputs['study'],rep,copy.deepcopy(case['case']['snapshot']))
   if rep.pos!=len(rep.folders) or sha(camp.canon(again))!=sha(camp.canon(trace)):raise ValueError('independent native trace or actual messages differ')
   native_exact=True
  actor=rt.CommonActor()
  for slot in sorted(folder.glob('call-*')):
   request=slot/'request.json'
   if not request.exists():continue
   q=read(request);actual=actor.project(q['native_messages']);body=camp.build_body(actual['messages'],config)
   if q['native_input_sha256']!=camp.message_sha(q['native_messages']) or q['input_sha256']!=actual['input_sha256'] or q['student_view_sha256']!=actual['input_sha256'] or q['body']!=body or q['body_sha256']!=sha(camp.canon(body)) or q['endpoint']!=config['endpoint']:raise ValueError('actual shared input/body/endpoint identity differs')
   claimed.append(q['global_new_claim'])
   if not (slot/'invoked.json').exists():
    if not (slot/'not-called.json').exists():raise ValueError('admitted source slot lacks invoke or explicit not-called evidence')
    continue
   sent=slot/'capture/request-body.bin'
   if sent.exists() and sent.read_bytes()!=camp.canon(body):raise ValueError('captured actual HTTP request bytes differ')
   outcome=slot/'outcome.json';r={};health=dict(wire_complete=False,raw_JSON_object=False,known_usage=False,response_not_repaired=True)
   if outcome.exists():
    o=read(outcome);r=o['result'];state={};raw=slot/'capture/response.sse'
    if raw.exists():
     try:parser.parse_sse(raw.read_bytes().splitlines(),config['model'],state=state)
     except Exception:pass
    if any(state.get(k)!=r.get(k) for k in ('content','usage','models','finish_reason','stream_done')):raise ValueError('independent original SSE parse differs')
    if o['provider_raw_content']!=r.get('content','') or o['provider_target_sha256']!=sha(o['provider_raw_content'].encode()):raise ValueError('provider target changed')
    health=wire.classify(o,config['model'])
   charged=recovery.charge(r,q['reservation_CNY']);cost+=charged['conservative_CNY'];unknown+=charged['unknown'];tokens[0]+=charged['input_tokens'];tokens[1]+=charged['output_tokens']
   calls.append(dict(source_index=case['source_index'],role=q['role'],stage=q['stage'],input_sha256=q['input_sha256'],same_information_exact=True,request=binding(request),outcome=binding(outcome) if outcome.exists() else None,SSE=binding(slot/'capture/response.sse') if (slot/'capture/response.sse').exists() else None,**health,**charged))
  legal=native_exact and trace.get('terminated') is True
  rows.append(dict(source_index=case['source_index'],source_id=case['source_id'],component=case['component'],group=case['group'],split=case['split'],missing=not native_exact,native_replay_exact=native_exact,legal=legal,stop_reason=trace.get('stop_reason'),prediction=trace.get('final_result') or {},category_correct=None,grade_correct=None,joint_correct=None,reference_scoring_pending=True,semantic_review_pending=True,executed_path=[s['call']['tool'] for p in trace.get('programs',[]) for s in p['steps']],first_program=[s['tool'] for s in trace['programs'][0]['plan']['program']] if trace.get('programs') else []))
 b=comp['budget']
 if claimed!=list(range(1,b['claims']+1)) or abs(cost-b['spent'])>1e-8 or unknown!=b['unknown'] or tokens!=[b['input_tokens'],b['output_tokens']]:raise ValueError('durable admitted/invoked/known/unknown whole-loop ledger differs')
 files=[binding(p) for p in target.rglob('*') if p.is_file() and p.resolve()!=receipt.resolve() and not p.name.endswith('.partial')]
 summary=dict(passed=True,actual_OS_exit=os_exit['actual_OS_exit'],fixed_denominator=len(inputs['cases']),completed=sum(r['native_replay_exact'] for r in rows),native_replays=sum(r['native_replay_exact'] for r in rows),legal=sum(r['legal'] for r in rows),actual_model_calls=len(calls),admitted_request_positions=len(claimed),unknown_usage=unknown,conservative_cost_CNY=cost,same_information_exact=all(c['same_information_exact'] for c in calls),wire_complete_all=all(c['wire_complete'] for c in calls),raw_JSON_objects=sum(c['raw_JSON_object'] for c in calls),reference_labels_loaded=False,reference_scoring_pending=True,semantic_verified_all=False,training_ready=False,TestOnly=testonly,new_external_API_by_this_verifier=0,new_GPU=0,time_unix=time.time())
 proof=dict(summary=summary,rows=rows,calls=calls,files=files,runtime_contract=contract());put(receipt,proof);return proof
