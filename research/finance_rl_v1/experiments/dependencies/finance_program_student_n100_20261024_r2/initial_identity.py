"""Restore the original canonical initial-state ordering, never alter source values."""
import json
def load_inputs(path):
 original=json.loads(path.read_bytes())
 canonical=json.loads(json.dumps(original,ensure_ascii=False,sort_keys=True,separators=(',',':')))
 assert canonical==original
 return canonical
