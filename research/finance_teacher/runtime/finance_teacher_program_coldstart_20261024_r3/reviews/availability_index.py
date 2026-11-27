"""CPU-only candidate; not imported by the current frozen live runtime."""
import copy,json

GUIDANCE=('\nobserved_acquisition_index_v1 groups only already visible observations by exact scope, target and attribute, '
 'across all producing Skills. A tool\'s empty own observations list means no recorded observation FROM THAT TOOL; '
 'it does not mean the corresponding target attribute has never been obtained by another producer. '
 'Obtained, canonical unavailable, error and unqueried are distinct. This index supplies no action, new value, '
 'readiness decision, category or sufficiency judgement. Use the original typed/availability/repeat/evidence guards unchanged.')

def _resolve(value,pool,seen=()):
 if isinstance(value,dict) and set(value)=={'$message_ref'}:
  key=value['$message_ref']
  if key in seen or key not in pool:raise ValueError('invalid public message reference')
  return _resolve(pool[key],pool,seen+(key,))
 if isinstance(value,dict):return {k:_resolve(v,pool,seen) for k,v in value.items()}
 if isinstance(value,list):return [_resolve(v,pool,seen) for v in value]
 return copy.deepcopy(value)

def build_index(body):
 if body.get('encoding')=='lossless-message-values-v1':root=body['payload'];pool=body['lossless_message_values']
 elif 'encoding' in body:raise ValueError('unknown public envelope; do not guess decoder')
 else:root=body;pool={}
 boundary=root.get('evidence_boundary') or {};raw=boundary.get('availability_ledger')
 if raw is None:raw=(root.get('current') or {}).get('named_state',{}).get('availability_ledger',[])
 ledger=_resolve(raw,pool)
 if not isinstance(ledger,list):raise ValueError('public ledger must be a list')
 groups={}
 for observation in ledger:
  if not isinstance(observation,dict) or any(not isinstance(observation.get(k),str) or not observation[k] for k in ('scope','type','status','skill')) or not isinstance(observation.get('target'),(str,list,dict)) or not observation['target']:raise ValueError('malformed public observation')
  key=(observation['scope'],json.dumps(observation['target'],ensure_ascii=False,sort_keys=True,separators=(',',':')),observation['type'])
  if key not in groups:groups[key]=dict(scope=observation['scope'],target=copy.deepcopy(observation['target']),type=observation['type'],observed_statuses=[],observed_producers=[],observations=[])
  row=groups[key];status=observation['status'];producer=observation['skill']
  if status not in row['observed_statuses']:row['observed_statuses'].append(status)
  if producer not in row['observed_producers']:row['observed_producers'].append(producer)
  pair=dict(status=status,producer=producer)
  if pair not in row['observations']:row['observations'].append(pair)
 return dict(schema='cross-producer-observed-target-attribute-v1',entries=list(groups.values()),evidence_added=False,automatic_actions=False,source_values_added=False,unknown_not_treated_as_unavailable=True,conflicting_statuses_not_resolved=True)

def rewrite(messages,actor):
 if actor not in ('student','teacher'):raise ValueError('unknown actor')
 out=copy.deepcopy(messages)
 if not out or out[0]['role']!='system':raise ValueError('trusted first system message required')
 if out[0]['content'].startswith('Execute only the selected Skill '):return out
 body=json.loads(out[-1]['content'])
 if 'observed_acquisition_index_v1' in body:raise ValueError('projection already applied')
 body['observed_acquisition_index_v1']=build_index(body);out[-1]['content']=json.dumps(body,ensure_ascii=False,separators=(',',':'));out[0]['content']+=GUIDANCE;return out
