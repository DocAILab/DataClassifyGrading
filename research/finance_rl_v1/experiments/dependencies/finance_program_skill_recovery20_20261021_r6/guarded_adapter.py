"""Per-case adapter; locally blocked requests never inherit real GPU evidence."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

path = Path(__file__).resolve().parent / 'failure_gate.py'
spec = importlib.util.spec_from_file_location('per_case_malformed_gate', path)
gate_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate_module)

class GuardedAdapter:
    def __init__(self, backend, rewriter):
        self.backend = backend
        self.rewriter = rewriter
        self.gate = gate_module.MalformedGate()
        self.last_generation_metadata = {}
        self.last_call_evidence = {}

    def generate(self, messages, arm, stage, destination):
        actual = self.rewriter(messages)
        self.last_generation_metadata = {}
        self.last_call_evidence = {}
        self.backend.last_generation_metadata = {}
        self.backend.last_call_evidence = {}
        try:
            return self.gate.run(stage, actual, lambda: self.backend.generate(actual, arm, stage, destination))
        except gate_module.RepeatedMalformedOutput as exc:
            self.last_call_evidence = dict(generation_invoked=False, generation_returned=False, locally_blocked=True)
            with (Path(destination) / 'local-block.json').open('x', encoding='utf-8') as stream:
                json.dump(dict(stage=stage, error=str(exc), **self.last_call_evidence), stream, ensure_ascii=False, indent=2)
            raise
        finally:
            self.last_generation_metadata = deepcopy(self.backend.last_generation_metadata)
            if not self.last_call_evidence.get('locally_blocked'):
                self.last_call_evidence = deepcopy(self.backend.last_call_evidence)
