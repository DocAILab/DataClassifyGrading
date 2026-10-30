"""Controller-owned argument transport; no semantic or execution decisions."""
from copy import deepcopy

from agent.orchestration.agent_loop import ActionError
from agent.orchestration.contracts import ToolCall
from agent.orchestration.validator import resolve_reference

BINDING_RULE_VERSION = 'canonical-state-bindings-v1'
EVIDENCE_SCHEMA_VERSION = 'typed-observed-evidence-v2'


def strict_equal(left,right):
    if type(left) is not type(right):
        return False
    if isinstance(right,dict):
        return left.keys()==right.keys() and all(strict_equal(left[k],v) for k,v in right.items())
    if isinstance(right,(list,tuple)):
        return len(left)==len(right) and all(strict_equal(a,b) for a,b in zip(left,right))
    return left==right


def bind_call(call,spec,record,state,binding_resolver):
    """Fill required state parameters; validate explicit values before normalization."""
    allowed=set(spec.required_inputs)|set(spec.optional_inputs)
    if call.tool!=spec.name or set(call.arguments)-allowed:
        raise ActionError('binding contains unknown Skill or arguments')
    normalized={}; changes=[]
    for key in (*spec.required_inputs,*spec.optional_inputs):
        supplied=key in call.arguments
        if call.tool=='ACQUIRE_MISSING_METADATA' and key=='missing_metadata_type':
            choice=call.arguments.get(key)
            if not isinstance(choice,str) or choice not in state.get('missing_metadata_types',[]):
                raise ActionError('missing_metadata_type choice must be an actual missing type')
            normalized[key]=choice
            continue
        reference=binding_resolver(key,record,state)
        if reference is None:
            if supplied or key in spec.required_inputs:
                raise ActionError(f'binding unavailable for {key}')
            continue
        found,value=resolve_reference(reference,record,state)
        if not found:
            raise ActionError(f'binding unresolved for {key}')
        if supplied:
            original=call.arguments[key]
            if not (type(original) is str and original==reference) and not strict_equal(original,value):
                raise ActionError(f'argument {key} differs from canonical binding {reference}')
            normalized[key]=reference
            if original!=reference:
                changes.append({'argument':key,'kind':'literal_to_reference','from':deepcopy(original),'to':reference})
        elif key in spec.required_inputs:
            normalized[key]=reference
            changes.append({'argument':key,'kind':'injected','to':reference})
    canonical=ToolCall(call.tool,normalized)
    return canonical,{'version':BINDING_RULE_VERSION,'raw_call':deepcopy(call.to_dict()),
                      'normalized_call':deepcopy(canonical.to_dict()),'changes':changes}
