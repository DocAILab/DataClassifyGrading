"""Same shared R3 messages; only the documented provider output mode changes."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PARENT=ROOT.parent/'finance_teacher_same_information_20261024_r3'
ANCHOR=ROOT.parent/'finance_teacher_same_information_20261024_r1'
sys.path.insert(0,str(PARENT))
from scope_contract import project_messages,read,put,sha,canon,message_sha,normalized_messages,binding
from src.guard import build_body as old_build_body,Budget,claim_once
from collect import load,verify_bindings,transport_critical_ok,TRANSPORT,CREDENTIALS,PARSER,RULES

def build_body(messages,config):
 body=old_build_body(messages,config)
 if not any('json' in x['content'].lower() for x in body['messages']):raise ValueError('JSON Object requires unchanged JSON instruction')
 body['response_format']=dict(type='json_object');return body

def audit_input(slot,q,config,base):
 base=normalized_messages(base);m=project_messages(base,'student')
 if slot['base_input_sha256']!=message_sha(base) or slot['messages']!=m or m!=project_messages(base,'teacher') or slot['input_sha256']!=message_sha(m) or q['input_sha256']!=slot['input_sha256'] or q['body']!=build_body(m,config) or q['body_sha256']!=sha(canon(q['body'])):raise ValueError('actual shared input/body/source differs')
 return True

def audit_target(o,r):
 content=r.get('content','')
 if not isinstance(content,str) or o['provider_raw_content']!=content or o['provider_target_sha256']!=sha(content.encode()):raise ValueError('provider target repaired or changed')
 return True

def audit_claims(claims):
 if claims!=list(range(1,len(claims)+1)):raise ValueError('claims missing or repeated')
 return True

def check_basic(proof,homogeneity):
 if proof.get('passed') is not True or proof.get('actual_OS_exit')!=0 or proof['structural_accepted']/proof['scheduled_slots']<.9 or not proof.get('all_actual_shared_messages_exact') or not proof.get('all_source_values_unchanged') or proof.get('partition_intersections')!=0 or homogeneity['exact_target_max_share']>.5 or homogeneity['SELECT_ordered_max_share']>.5:raise ValueError('basic identity/quality/diversity gate failed')
 return True

def make_admission(parent,claims,cost,new_cap):
 if claims!=112 or abs(cost-12.608544)>1e-8 or new_cap>112 or claims+new_cap>224 or parent['started_unix']!=1792831039.3679242 or parent['deadline_unix']!=1792845439.3679242:raise ValueError('campaign clock/ledger cannot reset')
 return dict(started_unix=parent['started_unix'],deadline_unix=parent['deadline_unix'],prior_claimed_HTTP=claims,prior_charged_CNY=cost,expansion_cap=new_cap,global_planned_HTTP=claims+new_cap,global_cost_cap_CNY=160,clock_not_reset=True)

def ledger(admission,budget):return dict(claimed=admission['prior_claimed_HTTP']+budget['claims'],charged_CNY=admission['prior_charged_CNY']+budget['spent'])
