"""Pure finite campaign and role-labelled cold-start identity checks."""
import copy,json,sys,time
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parent
STATIC=ROOT.parent/'finance_teacher_same_information_20261024_r4'
ANCHOR=ROOT.parent/'finance_teacher_same_information_20261024_r1'
R14=ROOT.parent/'finance_program_termination_repair_20261024_r14'
sys.path.insert(0,str(STATIC))
from format_runtime import read,put,sha,canon,message_sha,binding,verify_bindings,load,Budget,TRANSPORT,PARSER,CREDENTIALS
PRIVATE={'teacher_only_source_dossier','dossier','reference_label','expected_category','ground_truth','local_reference','source_table_id','source_id'}
def attempts(stopped):
 n=1
 while not stopped():yield n;n+=1
def remaining(attempt,new_steps):return dict(programs=None,planner_attempts=None,skill_steps=None,program_steps=None)
def audit_shared(student,teacher):
 if student!=teacher or not isinstance(student,list) or not student:raise ValueError('role actor views differ')
 def visit(v):
  if isinstance(v,dict):
   if set(v)&PRIVATE:raise ValueError('Teacher-only/private identity in prompt')
   for x in v.values():visit(x)
  elif isinstance(v,list):
   for x in v:visit(x)
 for m in student:
  if set(m)!={'role','content'} or not isinstance(m['content'],str):raise ValueError('message envelope')
  if 'teacher_only_source_dossier' in m['content']:raise ValueError('private dossier')
  if m['role']=='user':
   try:value=json.loads(m['content'])
   except ValueError:raise ValueError('expected structured native user body')
   visit(value)
 return True
def group_targets(rows):
 by=defaultdict(list)
 for row in rows:by[(row['role'],row['input_sha256'])].append(row)
 good=[];hold=[]
 for group in by.values():
  if len({r['target_sha256'] for r in group})>1:hold.extend(group)
  else:good.append(group[0])
 return good,hold
def audit_components(rows):
 seen={}
 for row in rows:
  key=row['component'];s=row['split']
  if key in seen and seen[key]!=s:raise ValueError('component leakage across role/episode pools')
  seen[key]=s
 return True
def new_admission(parent,started):
 if type(parent.get('claimed')) is not int or parent['claimed']<0 or parent['claimed']>224 or parent.get('charged_CNY',-1)<0 or parent['charged_CNY']>160:raise ValueError('invalid parent ledger')
 return dict(started_unix=started,deadline_unix=started+14400,new_HTTP_cap=400,new_cost_cap_CNY=80,parent_claimed_HTTP=parent['claimed'],parent_conservative_CNY=parent['charged_CNY'],old_clock_and_ledger_not_reset=True)

def continuation_admission(previous,completion,parent):
 if previous['new_HTTP_cap']!=400 or previous['new_cost_cap_CNY']!=80 or previous['deadline_unix']-previous['started_unix']!=14400:raise ValueError('old full campaign identity differs')
 b=completion['budget'];used=b['claims'];spent=b['spent']
 if not 0<=used<400 or not 0<=spent<80:raise ValueError('full campaign allowance exhausted')
 return dict(previous,remaining_HTTP_cap=400-used,remaining_cost_cap_CNY=80-spent,previous_full_claims=used,previous_full_conservative_CNY=spent,parent_claimed_HTTP=parent['claimed'],parent_conservative_CNY=parent['charged_CNY'],original_full_clock_not_reset=True,original_full_cost_not_reset=True)

FORMAT_SUFFIX='\n\nMachine output contract for both Student and Teacher: Return raw JSON only. Do not use Markdown, backticks, code fences, headings, or explanatory prose. The first non-whitespace character must be { and the last must be }. No text is permitted before or after the JSON object.'
def format_messages(messages,actor):
 if actor not in ('student','teacher'):raise ValueError('unknown actor')
 out=copy.deepcopy(messages)
 if not out or out[0]['role']!='system':raise ValueError('first system contract required')
 out[0]['content']+=FORMAT_SUFFIX;return out

def raw_json_object(content):
 try:return isinstance(json.loads(content),dict)
 except (ValueError,TypeError):return False

def cumulative_full_admission(original,history,parent):
 if original['new_HTTP_cap']!=400 or original['new_cost_cap_CNY']!=80 or original['deadline_unix']-original['started_unix']!=14400:raise ValueError('full authority identity changed')
 used=history['cumulative_full_claims'];spent=history['cumulative_full_conservative_CNY']
 if not 0<=used<400 or not 0<=spent<80:raise ValueError('full campaign allowance exhausted')
 return dict(original,remaining_HTTP_cap=400-used,remaining_cost_cap_CNY=80-spent,previous_full_claims=used,previous_full_conservative_CNY=spent,parent_claimed_HTTP=parent['claimed'],parent_conservative_CNY=parent['charged_CNY'],original_full_clock_not_reset=True,original_full_cost_not_reset=True)

def completed_revision_admission(original,previous,parent,now):
 if previous.get('actual_worker_exit')!=0 or previous.get('closed')!=10 or previous.get('fixed_denominator')!=10:raise ValueError('previous pilot not truly complete')
 if now>=original['deadline_unix']:raise ValueError('original full deadline expired')
 return cumulative_full_admission(original,previous,parent)

def static_ledger_snapshot():
 """Read only immutable old slots; live parent snapshot is explicitly not final."""
 complete=STATIC/'expansion-completion.json'
 if complete.exists():
  c=read(complete)
  if c['worker_exit_code']!=0:raise ValueError('static worker failed')
  return dict(ledger=c['global_ledger'],final=True,files=[binding(complete)])
 old=read(STATIC/'admission.json');claimed=0;cost=0.;files=[binding(STATIC/'admission.json')]
 for folder in sorted((STATIC/'collection/expansion').iterdir()):
  if not (folder/'invoked.json').exists():continue
  claimed+=1;q=read(folder/'request.json');files.extend([binding(folder/'invoked.json'),binding(folder/'request.json')]);o=folder/'outcome.json'
  try:result=read(o) if o.exists() else None
  except json.JSONDecodeError:result=None
  usage=result['result'].get('usage') if result else None
  if usage is None:cost+=q['reservation_CNY']
  else:
   cost+=(usage['prompt_tokens']*12+usage['completion_tokens']*36)/1000000
   files.append(binding(o))
 return dict(ledger=dict(claimed=old['prior_claimed_HTTP']+claimed,charged_CNY=old['prior_charged_CNY']+cost),final=False,files=files,includes_unknown_or_inflight_reservations=True)
def audit_initial(case):
 s=case['snapshot'];state=s['state']
 if set(state)!={'availability_ledger'} or s.get('prefix_calls')!=0 or s.get('prefix_steps') or s.get('feedback') is not None or s.get('counts') or any(s['lineage'].values()) or any(x.get('type')!='field_name' for x in state['availability_ledger']):raise ValueError('not canonical initial source-only state')
 return True
def select_cases(cases,slots):
 by={}
 for s in slots:
  sid=s['source_id'];v={k:s[k] for k in ('source_id','source_index','group','component','split')}
  if sid in by and by[sid]!=v:raise ValueError('source partition inconsistent')
  by[sid]=v
 selected=[];components=set()
 for case in sorted(cases,key=lambda x:sha(('full-teacher-20261024'+x['source_id']).encode())):
  sid=case['source_id'];meta=by.get(sid)
  if meta is None or meta['split']!='train' or meta['component'] in components:continue
  audit_initial(case);components.add(meta['component']);selected.append(dict(case=copy.deepcopy(case),**meta))
  if len(selected)==10:break
 if len(selected)!=10:raise ValueError('need exactly ten allowed train components; do not expand source pool')
 audit_components(selected);return selected
def build_body(messages,config):
 if not any('json' in m['content'].lower() for m in messages):raise ValueError('JSON instruction absent')
 return dict(model=config['model'],messages=copy.deepcopy(messages),temperature=0,max_tokens=8192,thinking_budget=2048,enable_thinking=True,enable_search=False,stream=True,stream_options=dict(include_usage=True),response_format=dict(type='json_object'))
def runtime_dependencies(root,freeze):
 deps=[];archived=[]
 for key,base in (('files',Path(root)),('dependencies',Path(root).parent)):
  for row in freeze[key]:
   p=base/row['path']
   if not p.is_file():
    if '/runs/' not in row['path'] or p.suffix=='.py':raise FileNotFoundError('missing executable runtime or input dependency')
    archived.append(row);continue
   b=binding(p)
   if b['bytes']!=row['bytes'] or b['sha256']!=row['sha256']:raise ValueError('present dependency differs from old SHA')
   deps.append(b)
 return deps,archived
