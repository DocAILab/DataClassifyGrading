"""Read-only summary of unexecuted model intent; never execution evidence."""
import json
import math
from .validation import _unique_object

def _reject_constant(value):
    raise ValueError(f'nonfinite JSON {value}')

def _finite_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError('nonfinite JSON number')
    return number

def rejected_program_summary(raw, catalog, max_steps):
    if not isinstance(raw, str):
        return None
    try:
        value = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_reject_constant, parse_float=_finite_float)
    except (ValueError, TypeError, RecursionError):
        return None
    if not isinstance(value, dict) or set(value) != {'program'}:
        return None
    steps = value['program']
    if not isinstance(steps, list) or not steps:
        return None
    tools = []
    for item in steps:
        if not isinstance(item, dict) or set(item) != {'tool','arguments'}:
            return None
        name = item['tool']
        if not isinstance(name, str) or name not in catalog or not isinstance(item['arguments'], dict):
            return None
        tools.append(name)
    return {'executed':False,'trusted':False,'step_count':len(steps),
            'max_program_steps':max_steps,'tool_sequence':tools}
