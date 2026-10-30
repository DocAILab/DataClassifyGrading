"""Only prevent exact repeats of a syntactically malformed synthesis request."""
import hashlib
import json

class RepeatedMalformedOutput(RuntimeError):
    pass

class MalformedGate:
    def __init__(self):
        self.failed = {}

    def run(self, stage, messages, generate):
        if stage != 'SYNTHESIZE_METADATA_SEMANTICS':
            return generate()
        key = hashlib.sha256(json.dumps(dict(stage=stage, messages=messages), ensure_ascii=False,
            separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()
        if key in self.failed:
            raise RepeatedMalformedOutput('identical_malformed_synthesis_request_blocked: ' + key +
                '; acquire relevant new evidence or choose a different valid route; prior raw output SHA ' + self.failed[key])
        raw = generate()
        try:
            json.loads(raw)
        except json.JSONDecodeError:
            self.failed[key] = hashlib.sha256(raw.encode('utf-8')).hexdigest()
        # Preserve original bytes, including invalid JSON, for native validation.
        return raw
