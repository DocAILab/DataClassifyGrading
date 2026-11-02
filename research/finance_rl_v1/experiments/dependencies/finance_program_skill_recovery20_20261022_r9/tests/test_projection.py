import copy
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('new_contract_projection', ROOT/'interface.py')
interface = importlib.util.module_from_spec(spec)
spec.loader.exec_module(interface)
OLD = ROOT.parent/'finance_program_skill_recovery20_20261022_r8/reviews/final-01/extract/runs'

class ProjectionTests(unittest.TestCase):
    def test_real_missing_dependency_exposes_exact_existing_producer(self):
        trace = json.loads((OLD/'native-07-unified/trace.json').read_bytes())
        messages = trace['planner_attempts'][2]['messages']
        result = interface.rewrite(messages)
        marker = '\nLocal dependency projection (not an executed plan):\n'
        self.assertIn(marker, result[0]['content'])
        data = json.loads(result[0]['content'].split(marker)[-1])
        self.assertEqual(data['missing_dependency'], 'table_semantics')
        self.assertIn('INTERPRET_TABLE_NAME', [p['name'] for p in data['declared_producers']])
        self.assertNotIn('INTERPRET_TABLE_DESCRIPTION', [p['name'] for p in data['declared_producers']])
        self.assertEqual(data['evidence_added'], False)

    def test_real_classification_missing_key_has_complete_template(self):
        messages = json.loads((OLD/'native-15-unified/call-014/started.json').read_bytes())['messages']
        result = interface.rewrite(messages)
        marker = '\nComplete output shapes (structure only; replace every placeholder):\n'
        self.assertIn(marker, result[0]['content'])
        data = json.loads(result[0]['content'].split(marker)[-1])
        for example in data:
            self.assertEqual(set(example), {'option_id','summary','support'})
            self.assertEqual(set(example['support']), {'sufficient','citations','missing_evidence'})
            self.assertIsInstance(example['support']['missing_evidence'], str)

    def test_real_semantic_output_has_both_objects_closed(self):
        messages = json.loads((OLD/'native-15-unified/call-009/started.json').read_bytes())['messages']
        result = interface.rewrite(messages)
        marker = '\nComplete output shape (structure only; replace every placeholder):\n'
        self.assertIn(marker, result[0]['content'])
        example = json.loads(result[0]['content'].split(marker)[-1])
        self.assertEqual(set(example), {'summary','data'})
        self.assertEqual(set(example['data']), {'metadata_semantics'})

    def test_user_facts_and_original_object_not_modified(self):
        messages = json.loads((OLD/'native-15-unified/call-014/started.json').read_bytes())['messages']
        before = copy.deepcopy(messages)
        actual = interface.rewrite(messages)
        previous = interface.previous.rewrite(messages)
        self.assertEqual(messages, before)
        self.assertEqual(actual[-1], previous[-1])

if __name__=='__main__': unittest.main()
