"""Compare saved JSON traces independently of object-key serialization order."""
import json


def same_trace(actual, expected):
    options = dict(ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return json.dumps(actual, **options).encode('utf-8') == json.dumps(expected, **options).encode('utf-8')
