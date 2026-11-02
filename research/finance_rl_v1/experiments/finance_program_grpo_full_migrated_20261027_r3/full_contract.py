import hashlib,json

def schedule(ids,heldout):
 if len(ids)!=48 or len(set(ids))!=48 or set(ids)&set(heldout):raise ValueError('all48 unique allowed train IDs, no evaluation targets')
 ordered=sorted(ids,key=lambda x:hashlib.sha256(('full48-20261027:'+x).encode()).hexdigest())
 return [ordered[i:i+8] for i in range(0,48,8)]

def groups(rows,policy_version):
 by={}
 for v in rows:
  if v['policy_version']!=policy_version or v['reward']['A1']!=float(v['reward']['outcome']):raise ValueError('current policy and pure joint reward required')
  by.setdefault(v['source_id'],[]).append(v)
 if len(rows)!=32 or len(by)!=8:raise ValueError('eight fixed complete groups')
 for gg in by.values():
  if sorted(x['sample_index'] for x in gg)!=list(range(4)):raise ValueError('unique four samples')
 return list(by.values())

def gate(rows,policy_version):
 g=groups(rows,policy_version);mixed=sum(len({v['reward']['outcome'] for v in gg})==2 for gg in g)
 legal=sum(v['legal_completion'] for v in rows);exact=all(v['native_replay_exact'] and v['actual_initial_exact'] for v in rows)
 return dict(episodes=32,sources=8,mixed_sources=mixed,legal_predictions=legal,all_native_exact=exact,allow_update=legal>=30 and exact and mixed>=1,reward='A1',maximum_updates=6)

def ensure_update_allowed(backend):
 if backend.admission['maximum_optimizer_updates']!=6 or not 0<=backend.version<6:raise ValueError('six update limit')
 return 6
