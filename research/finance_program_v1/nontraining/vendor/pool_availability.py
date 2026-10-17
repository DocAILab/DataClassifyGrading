"""Record-local query evidence. Never accesses the private metadata lookup."""
from copy import deepcopy
from pool_metadata import FIELDS, present

RULE = ('observed-query-availability-v1: unknown permits acquisition; only canonical-val '
        'query observations establish obtained or unavailable; errors remain retryable; '
        'reuse obtained values; suppress repeat targets in the same record/object scope; '
        'CHECK reports visibility and omitted batch fields stay unknown; missing_metadata_types '
        'retains its acquired-visibility meaning; continue evidence and classification when sufficient.')
DIRECT = {'GET_FIELD_DESCRIPTION':'field_description', 'GET_TABLE_DESCRIPTION':'table_description',
          'GET_DATABASE_DESCRIPTION':'database_description'}
CONTEXT = {'GET_CONTEXT_FIELD_DESCRIPTIONS':('context_field_descriptions','field_description'),
           'GET_CONTEXT_FIELD_TYPES':('context_field_types','field_type'),
           'GET_CONTEXT_FIELD_VALUES':('context_field_values','value')}
BATCH = {'GET_FIELD_METADATA':'field_metadata', 'GET_TABLE_FIELDS':'table_fields',
         'GET_COMPLETE_TABLE_CONTEXT':'table_context', 'GET_COMPLETE_DATABASE_CONTEXT':'database_context',
         'EXPAND_SELECTED_CONTEXT':'context_fields'}


def initialize(record, state):
    state.setdefault('availability_ledger', [{'scope':'record','target':record.field_name,
        'type':'field_name','status':'obtained','skill':'initial-input','source':'public-input',
        'value':record.field_name}])


def status(state, scope, target, kind):
    return next((e['status'] for e in state.get('availability_ledger',[])
        if (e['scope'],e['target'],e['type']) == (scope,target,kind)), 'unknown')


def target_name(state):
    return next(e['value'] for e in state['availability_ledger']
        if e['scope']=='record' and e['type']=='field_name')


def targets(call, state):
    name=call.tool; own=target_name(state)
    if name in DIRECT: return [('record',own,DIRECT[name])]
    if name=='ACQUIRE_MISSING_METADATA': return [('record',own,call.arguments.get('missing_metadata_type'))]
    if name in BATCH:
        # Expansion depends on the selected context object set, not a global tool flag.
        target=own if name!='EXPAND_SELECTED_CONTEXT' else tuple(state.get('selected_context_fields',[]))
        return [('query',list(target) if isinstance(target,tuple) else target,BATCH[name])]
    if name in CONTEXT:
        return [('context-field',f['field_name'],CONTEXT[name][1]) for f in state.get('context_fields',[])]
    return []


def blocked(call, state):
    queried=targets(call,state)
    return queried if queried and all(status(state,*t) in ('obtained','unavailable') for t in queried) else []


def _put(state, scope, target, kind, outcome, skill, source, value=None):
    entry={'scope':scope,'target':deepcopy(target),'type':kind,'status':outcome,'skill':skill,'source':source}
    if outcome=='obtained': entry['value']=deepcopy(value)
    ledger=state['availability_ledger']
    for i,old in enumerate(ledger):
        if (old['scope'],old['target'],old['type'])==(scope,target,kind):
            if old['status'] in ('obtained','unavailable') and outcome=='error': return
            ledger[i]=entry;break
    else: ledger.append(entry)


def observe(call, observation, state):
    if call.tool=='CHECK_METADATA_AVAILABILITY': return
    queried=targets(call,state)
    if not queried: return
    source=observation.metadata.get('source'); outcome=observation.status
    if outcome=='error':
        for t in queried: _put(state,*t,'error',call.tool,None)
        return
    if source!='canonical-val': return
    own=target_name(state);data=observation.data
    if call.tool in CONTEXT:
        wrapper=data.get(CONTEXT[call.tool][0],{})
        available={f['field_name']:f[CONTEXT[call.tool][1]] for f in wrapper.get('fields',[])
                   if present(f.get(CONTEXT[call.tool][1]))}
        missing=wrapper.get('missing_fields',[]) if outcome=='success' else observation.metadata.get('missing_fields',[])
        for scope,target,kind in queried:
            if outcome=='success' and target in available: _put(state,scope,target,kind,'obtained',call.tool,source,available[target])
            elif target in missing: _put(state,scope,target,kind,'unavailable',call.tool,source)
        return
    if outcome=='unavailable':
        for t in queried:
            if t[0]=='record' and t[2] in observation.metadata.get('missing',[]):
                _put(state,*t,'unavailable',call.tool,source)
        return
    if outcome!='success': return
    raw={}
    for key in ('field_metadata','acquired_metadata'):
        raw.update({k:v for k,v in data.get(key,{}).items() if k in FIELDS and present(v)})
    raw.update({k:v for k,v in data.items() if k in FIELDS and present(v)})
    if 'field_name' in raw and raw['field_name']!=own: raise ValueError('observed field_name differs from initial public input')
    # JSON snapshots sort mapping keys; canonical ledger order must survive that
    # round trip because the next Planner payload includes this ordered list.
    for k,v in sorted(raw.items()): _put(state,'record',own,k,'obtained',call.tool,source,v)
    if call.tool in BATCH:
        for t in queried: _put(state,*t,'obtained',call.tool,source,data.get(BATCH[call.tool]))
    if call.tool=='EXPAND_SELECTED_CONTEXT':
        for field in data.get('context_fields',[]):
            for key in FIELDS:
                if present(field.get(key)): _put(state,'context-field',field['field_name'],key,'obtained',call.tool,source,field[key])


def filter_callable(state, catalog):
    result={}
    from agent.orchestration.contracts import ToolCall
    for name,spec in catalog.items():
        if name=='ACQUIRE_MISSING_METADATA':
            if not any(not blocked(ToolCall(name,{'missing_metadata_type':key}),state)
                       for key in state.get('missing_metadata_types',[])): continue
        elif blocked(ToolCall(name,{}),state): continue
        result[name]=spec
    return result
