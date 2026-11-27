"""Finite technical recovery; no prompt, evidence or response modifications."""
import collections,ctypes,math,os

def allowance(original,history,now):
 if (original.get('new_HTTP_cap'),original.get('new_cost_cap_CNY'))!=(400,80) or original['started_unix']!=1792842805.4330828 or original['deadline_unix']!=1792857205.4330828:raise ValueError('original authority changed')
 used=history['cumulative_full_claims'];spent=history['cumulative_full_conservative_CNY']
 if type(used) is not int or not 0<=used<400 or not math.isfinite(spent) or not 0<=spent<80 or now>=original['deadline_unix']:raise ValueError('original finite allowance exhausted')
 return dict(started_unix=original['started_unix'],deadline_unix=original['deadline_unix'],previous_HTTP=used,previous_CNY=spent,remaining_HTTP=400-used,remaining_CNY=80-spent,unknown_reservations_inherited=True)

def unstarted(cases,done):
 ids=[r['source_index'] for r in done];allids=[r['source_index'] for r in cases]
 if len(ids)!=len(set(ids)) or len(allids)!=len(set(allids)) or not set(ids)<=set(allids):raise ValueError('source identity mismatch')
 return [r for r in cases if r['source_index'] not in ids]

def charge(result,reservation):
 if not isinstance(reservation,(int,float)) or not math.isfinite(reservation) or reservation<0:raise ValueError('invalid reservation')
 u=result.get('usage');known=isinstance(u,dict) and all(type(u.get(k)) is int and u[k]>=0 for k in ('prompt_tokens','completion_tokens','total_tokens')) and u['prompt_tokens']+u['completion_tokens']==u['total_tokens']
 return dict(conservative_CNY=(u['prompt_tokens']*12+u['completion_tokens']*36)/1e6 if known else reservation,unknown=0 if known else 1,input_tokens=u['prompt_tokens'] if known else 0,output_tokens=u['completion_tokens'] if known else 0)

def basic(old,new,technical):
 rows=old+new;legal=sum(r['legal'] for r in rows);correct=sum(r['category_correct'] for r in rows)
 names={'INFER_CATEGORY':'INFER_CATEGORY_FROM_TOPK'};names.update({n:'SELECT_CATEGORY_TOPK' for n in ('SEARCH_STANDARD','SEARCH_STANDARD_BY_FIELD','SEARCH_STANDARD_BY_CONTEXT','SEARCH_STANDARD_BY_TABLE','SEARCH_STANDARD_BY_DATABASE','RESEARCH_STANDARD')})
 paths=collections.Counter(tuple(names.get(n,n) for n in r['executed_path']) for r in rows);ratio=max(paths.values(),default=0)/max(1,len(rows))
 criteria=dict(fixed10_unique=len(rows)==10 and len({r['source_index'] for r in rows})==10,all10_nativeexact=all(r.get('native_replay_exact') is True for r in rows),legal_at_least_half=legal>=5,correct_majority=correct>=3 and correct*2>legal,new_actual_OS0=technical.get('actual_OS_exit')==0 and technical.get('completed')==2,new_wire_healthy=technical.get('all_new_wire_complete') is True,sameinfo=technical.get('all_same_information_exact') is True,JSON_at_least90=technical.get('raw_JSON_fraction',0)>=.9,path_not_severe=ratio<=.8,historical_unknown_explicitly_preserved=technical.get('historical_unknown')==1 and technical.get('historical_unknown_preserved') is True)
 return dict(passed=all(criteria.values()),criteria=criteria,legal=legal,category_correct=correct,fixed_denominator=10,normalized_path_max_ratio=ratio,normalized_paths=[dict(path=list(k),count=n) for k,n in paths.most_common()],old_native=8,new_native=2,semantic_verified_all=False,training_ready=False)

def wait_seconds(wall_deadline,mono_deadline,wall_now,mono_now):return max(0.,min(.25,wall_deadline-wall_now,mono_deadline-mono_now))

class ActiveWork:
 """Thread-scoped idle-sleep request only; explicit lid/button remains honored."""
 def __init__(self,setter=None):self.setter=setter;self.previous=None
 def __enter__(self):
  if self.setter is None:
   if os.name!='nt':return self
   kernel=ctypes.WinDLL('kernel32',use_last_error=True);fn=kernel.SetThreadExecutionState;fn.argtypes=[ctypes.c_uint];fn.restype=ctypes.c_uint;self.setter=fn
  self.previous=self.setter(0x80000001)
  if not self.previous:raise OSError('bounded active-work execution request failed')
  return self
 def __exit__(self,*exc):
  if self.previous is not None and not self.setter(self.previous):raise OSError('execution state restoration failed')
