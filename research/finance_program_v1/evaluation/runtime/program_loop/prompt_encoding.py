"""Lossless repeated-observation references, not execution/state references."""
from collections import Counter
from copy import deepcopy
import json

SCOPE = ('current', 'feedback', 'recovery_context')
TAG = 'prompt_ref'
TABLE = 'shared_values'
SCHEMA = 'prompt_encoding_schema'


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def pack_prompt(payload):
    if not isinstance(payload, dict) or TABLE in payload or SCHEMA in payload:
        raise ValueError('reserved prompt encoding fields already present')
    counts = Counter()
    originals = {}

    def count(value):
        if isinstance(value, dict) and set(value) == {TAG}:
            raise ValueError('reserved prompt reference collision')
        if isinstance(value, (dict, list, str)):
            key = _canonical(value)
            if len(key) >= 256:
                counts[key] += 1
                originals[key] = value
            children = value.values() if isinstance(value, dict) else value if isinstance(value, list) else ()
            for child in children:
                count(child)

    for name in SCOPE:
        if name in payload:
            count(payload[name])
    refs = {}
    values = {}

    def walk(value, excluded=None):
        key = _canonical(value) if isinstance(value, (dict, list, str)) else ''
        if counts[key] > 1 and key != excluded:
            if key not in refs:
                name = 'v' + str(len(refs) + 1)
                refs[key] = name
                values[name] = walk(originals[key], key)
            return {TAG: refs[key]}
        if isinstance(value, dict):
            return {name: walk(child) for name, child in value.items()}
        if isinstance(value, list):
            return [walk(child) for child in value]
        return value

    encoded = deepcopy(payload)
    for name in SCOPE:
        if name in encoded:
            encoded[name] = walk(encoded[name])
    encoded[SCHEMA] = 1
    encoded[TABLE] = values
    if _canonical(unpack_prompt(encoded)) != _canonical(payload):
        raise ValueError('lossless prompt roundtrip failed')
    return encoded


def unpack_prompt(encoded):
    if not isinstance(encoded, dict) or encoded.get(SCHEMA) != 1 or not isinstance(encoded.get(TABLE), dict):
        raise ValueError('invalid prompt encoding schema')
    values = encoded[TABLE]
    used = set()
    active = set()
    cache = {}

    def expand(value):
        if isinstance(value, dict) and set(value) == {TAG}:
            name = value[TAG]
            if not isinstance(name, str) or name not in values or name in active:
                raise ValueError('dangling or cyclic prompt reference')
            used.add(name)
            if name not in cache:
                active.add(name)
                cache[name] = expand(values[name])
                active.remove(name)
            return deepcopy(cache[name])
        if isinstance(value, dict):
            return {name: expand(child) for name, child in value.items()}
        if isinstance(value, list):
            return [expand(child) for child in value]
        return value

    result = {name: deepcopy(value) for name, value in encoded.items() if name not in (TABLE, SCHEMA)}
    for name in SCOPE:
        if name in result:
            result[name] = expand(result[name])
    if used != set(values):
        raise ValueError('unused shared value could conceal unobserved evidence')
    return result
