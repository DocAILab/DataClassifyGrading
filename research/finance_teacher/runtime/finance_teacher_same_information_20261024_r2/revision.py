"""Additive shared contract and interruption accounting; no credential handling."""
import copy,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PARENT=ROOT.parent/'finance_teacher_same_information_20261024_r1'
sys.path.insert(0,str(PARENT))
from src.guard import read,put,sha,canon,message_sha,normalized_messages

COMMON_POLICY='''
Shared grounding clarification v2 (identical for Student and Teacher):
Before emitting sufficient=true, privately identify the necessary scope distinction in the supplied definition and check that actually acquired source text establishes it, not merely a word appearing in its examples. A generic attribute can occur in several business contexts. Do not silently assume an unstated market, product, customer, institution, or internal-enterprise relationship. A source quote establishes what the source says, not additional premises required by the category definition. A hypothesis or an abbreviation cannot supply those premises. A residual/other category also has a positive scope; absence of a product label does not prove internal enterprise management. Conversely, do not require extra physical types, sample values or nonexistent metadata when the acquired field/table purpose already establishes the relevant scope. Use genuinely acquired same-table descriptions to resolve the missing distinction, assess contrary facts, and cite the exact supporting eligible fact IDs. If no supplied option has the necessary scope established, preserve UNRESOLVED with a specific missing distinction, not a guessed class or a general demand for all metadata. This is a definition-consistency rule for every candidate, not a preferred class or a new source fact.
Final output must be the single JSON instance itself. Do not wrap it in Markdown code fences, return a schema, or return an array of examples. Native candidate, typed, exact-citation, persistent-gap and terminal guards are unchanged.
'''

def project_messages(messages,actor):
    if actor not in ('student','teacher'):raise ValueError('only common Student/Teacher projection')
    out=copy.deepcopy(normalized_messages(messages));out[0]['content']+=COMMON_POLICY
    return normalized_messages(out)

def interruption_row(folder):
    folder=Path(folder);request=read(folder/'request.json');out=read(folder/'outcome.json') if (folder/'outcome.json').exists() else None
    usage=out.get('result',{}).get('usage') if out else None
    valid=isinstance(usage,dict) and all(type(usage.get(k)) is int and usage[k]>=0 for k in ('prompt_tokens','completion_tokens','total_tokens')) and usage['total_tokens']==usage['prompt_tokens']+usage['completion_tokens']
    invoked=(folder/'invoked.json').exists()
    cost=usage['prompt_tokens']*12/1e6+usage['completion_tokens']*36/1e6 if valid else request['reservation_CNY'] if invoked else 0.
    return dict(id=folder.name,state=('returned_accepted' if out.get('admitted_transport') else 'returned_rejected') if out else 'interrupted_unknown_usage' if invoked else 'not_called',invoked=invoked,charged_CNY=cost,unknown_usage=int(invoked and not valid),actual_OS_exit=None,retry_allowed=False)

def admit_revision(admission,prior_claims,prior_cost,pilot,expansion):
    if time.time()>=admission['deadline_unix'] or prior_claims+pilot+expansion>224 or prior_cost>=160:raise ValueError('original finite campaign exhausted')
    return dict(started_unix=admission['started_unix'],deadline_unix=admission['deadline_unix'],prior_claimed_HTTP=prior_claims,prior_charged_CNY=prior_cost,pilot_slots=pilot,expansion_cap=expansion,global_planned_HTTP=prior_claims+pilot+expansion,global_cost_cap_CNY=160,clock_not_reset=True)

def detached_flags():return 0x8|0x200|0x01000000

def binding(path):
    p=Path(path);raw=p.read_bytes();return dict(path=str(p.resolve()),sha256=sha(raw),bytes=len(raw))

def prepare():
    from collect import load,PARSER,verify_bindings
    parent_freeze=read(PARENT/'freeze-01.json');verify_bindings(parent_freeze['files'])
    parser=load('interruption_independent_parser',PARSER);old=read(PARENT/'schedule.json');config=read(PARENT/'config.json');rows=[];files=[]
    for s in old['slots']:
        if s['phase']!='pilot':continue
        p=PARENT/'collection/pilot'/s['id']
        if not (p/'request.json').exists():continue
        row=interruption_row(p);state={};err=None
        try:parser.parse_sse((p/'capture/response.sse').read_bytes().splitlines(),config['model'],state=state)
        except Exception as exc:err=type(exc).__name__
        row.update(parser_error=err,stream_done=state.get('stream_done'),finish_reason=state.get('finish_reason'),parsed_complete=state.get('complete'),request_sha256=sha((p/'request.json').read_bytes()))
        if (p/'outcome.json').exists():
            out=read(p/'outcome.json')
            if state.get('content','')!=out['provider_raw_content'] or state.get('usage')!=out['result'].get('usage'):raise ValueError('independent SSE content/usage mismatch')
        rows.append(row);files.extend(binding(f) for f in p.rglob('*') if f.is_file())
    audit=dict(created_unix=time.time(),parent_freeze_sha256=sha((PARENT/'freeze-01.json').read_bytes()),actual_OS_exit=None,root_cause_confirmed=False,observed_owned_PIDs_absent=True,observed_exec_session_absent=True,claimed=len(rows),returned=sum(x['state'].startswith('returned') for x in rows),accepted=sum(x['state']=='returned_accepted' for x in rows),unknown_usage=sum(x['unknown_usage'] for x in rows),charged_CNY=sum(x['charged_CNY'] for x in rows),rows=rows,files=files,partial_slot_not_retried=True)
    # No retrospective normal-completion/OS-exit record is manufactured.
    put(PARENT/'reviews/interruption-evidence-01.json',audit)
    admission=admit_revision(read(PARENT/'admission.json'),len(rows),audit['charged_CNY'],50,142)
    config.update(pilot_HTTP_cap=50,pilot_cost_cap_CNY=60-audit['charged_CNY'],expansion_HTTP_cap=142,expansion_cost_cap_CNY=100,original_deadline_unix=admission['deadline_unix'])
    projected=[]
    for s in old['slots']:
        if s['phase']=='expansion' and sum(q['phase']=='expansion' for q in projected)>=142:continue
        t=copy.deepcopy(s);base=t.pop('actual_record');m=project_messages(s['messages'],'student')
        if m!=project_messages(s['messages'],'teacher') or m[1:]!=s['messages'][1:]:raise ValueError('shared projection changed source facts')
        t.update(base_actual_record=base,base_input_sha256=s['input_sha256'],messages=m,input_sha256=message_sha(m),contract_sha256=sha(m[0]['content'].encode()),common_projection='shared-grounding-v2');projected.append(t)
    dependencies=list(parent_freeze['files'])+[binding(PARENT/'freeze-01.json'),binding(PARENT/'schedule.json'),binding(PARENT/'admission.json'),binding(PARENT/'reviews/interruption-evidence-01.json')]
    put(ROOT/'config.json',config);put(ROOT/'schedule.json',dict(slots=projected,parent_schedule_sha256=sha((PARENT/'schedule.json').read_bytes()),original_selection_and_split_unchanged=True,not_original_message_byte_exact=True,shared_projected_messages_exact=True))
    put(ROOT/'admission.json',admission);put(ROOT/'dependencies.json',dependencies)
    print(json.dumps(dict(parent_claimed=len(rows),parent_returned=audit['returned'],parent_accepted=audit['accepted'],prior_unknown_usage=audit['unknown_usage'],prior_charged_CNY=audit['charged_CNY'],new_scheduled=len(projected),deadline_unix=admission['deadline_unix'],new_HTTP=0)))

if __name__=='__main__':prepare()
