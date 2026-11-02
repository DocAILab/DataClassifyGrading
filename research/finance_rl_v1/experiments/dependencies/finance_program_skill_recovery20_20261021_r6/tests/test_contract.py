import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT.parent / 'finance_program_skill_recovery20_20261021_r5'
CASE = OLD / 'reviews/final-01/extract/runs/native-17-unified'

def module(name):
    spec = importlib.util.spec_from_file_location('repair_' + name, ROOT / (name + '.py'))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result

class ContractTests(unittest.TestCase):
    def test_target_prompt_short_complete_shape_and_other_skill_unchanged(self):
        interface = module('interface')
        original = json.loads((CASE / 'call-009/returned.json').read_bytes())['messages']
        actual = interface.rewrite(original)
        baseline = interface.previous.rewrite(original)
        self.assertIn('Close BOTH JSON objects', actual[0]['content'])
        self.assertIn('100 words', actual[0]['content'])
        self.assertEqual(actual[1:], baseline[1:])
        unrelated = [dict(role='system', content='Execute only the selected Skill INTERPRET_FIELD_NAME.'), dict(role='user', content='{}')]
        self.assertEqual(interface.rewrite(unrelated), interface.previous.rewrite(unrelated))

    def gate(self):
        gate = module('failure_gate')
        self.assertTrue(hasattr(gate, 'MalformedGate'), 'missing protection for identical malformed source request')
        return gate, gate.MalformedGate()

    def test_identical_real_failure_not_regenerated_or_repaired(self):
        gate_module, gate = self.gate()
        responses = [json.loads((CASE / f'call-{i:03d}/model-returned.json').read_bytes()) for i in (9, 14, 17)]
        self.assertEqual(len({r['raw_text'] for r in responses}), 1)
        messages = json.loads((CASE / 'call-009/model-request.json').read_bytes())['messages']
        calls = []
        def generate():
            calls.append(True)
            return responses[0]['raw_text']
        raw = gate.run('SYNTHESIZE_METADATA_SEMANTICS', messages, generate)
        self.assertEqual(raw, responses[0]['raw_text'])
        with self.assertRaises(json.JSONDecodeError):
            json.loads(raw)
        with self.assertRaisesRegex(gate_module.RepeatedMalformedOutput, 'acquire relevant new evidence or choose a different valid route'):
            gate.run('SYNTHESIZE_METADATA_SEMANTICS', messages, generate)
        self.assertEqual(len(calls), 1)

    def test_changed_input_is_not_blocked(self):
        _, gate = self.gate()
        gate.run('SYNTHESIZE_METADATA_SEMANTICS', [{'content': 'old'}], lambda: '{')
        valid = '{"summary":"observed","data":{"metadata_semantics":"new evidence"}}'
        self.assertEqual(gate.run('SYNTHESIZE_METADATA_SEMANTICS', [{'content': 'new'}], lambda: valid), valid)

    def test_valid_repeat_and_non_target_stage_unchanged(self):
        _, gate = self.gate()
        valid = '{"summary":"observed","data":{"metadata_semantics":"provided facts only"}}'
        for _ in range(2):
            self.assertEqual(gate.run('SYNTHESIZE_METADATA_SEMANTICS', [], lambda: valid), valid)
            self.assertEqual(gate.run('Planner', [], lambda: '{'), '{')

    def test_adapter_clears_model_metadata_when_local_guard_blocks(self):
        adapter_module = module('guarded_adapter')
        self.assertTrue(hasattr(adapter_module, 'GuardedAdapter'), 'gate not wired into runtime adapter')
        class Delegate:
            last_generation_metadata = {}
            last_call_evidence = {}
            calls = 0
            def generate(self, messages, arm, stage, destination):
                self.calls += 1
                self.last_generation_metadata = {'input_tokens': 1}
                self.last_call_evidence = {'generation_invoked': True, 'generation_returned': True}
                return '{'
        delegate = Delegate()
        adapter = adapter_module.GuardedAdapter(delegate, lambda x: x)
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            first, second = Path(tmp) / 'first', Path(tmp) / 'second'
            first.mkdir(); second.mkdir()
            adapter.generate([], 'B', 'SYNTHESIZE_METADATA_SEMANTICS', first)
            with self.assertRaises(RuntimeError):
                adapter.generate([], 'B', 'SYNTHESIZE_METADATA_SEMANTICS', second)
            self.assertEqual(delegate.calls, 1)
            self.assertEqual(adapter.last_generation_metadata, {})
            self.assertEqual(adapter.last_call_evidence['generation_invoked'], False)
            self.assertFalse((second / 'model-returned.json').exists())
            blocked = json.loads((second / 'local-block.json').read_bytes())
            self.assertFalse(blocked['generation_invoked'])
            self.assertFalse(blocked['generation_returned'])

if __name__ == '__main__':
    unittest.main()
