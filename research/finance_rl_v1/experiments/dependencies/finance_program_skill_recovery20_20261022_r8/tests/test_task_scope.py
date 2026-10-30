import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

candidate = ROOT / 'interface.py'
if not candidate.exists():
    candidate = ROOT.parent / 'finance_program_skill_recovery20_20261021_r7/interface.py'
interface = load(candidate, 'scope_candidate')
codec = load(ROOT.parent / 'finance_program_skill_recovery20_20261021_r4/interface.py', 'original_codec')
boundary = load(ROOT.parent / 'finance_program_student_b80_eval_20261021_r2/vendor/finance_program_teacher_evidence_20261020_r1/boundary.py', 'original_strict_boundary')

class ScopeTests(unittest.TestCase):
    def test_classifier_distinguishes_field_definition_from_row_values(self):
        payload = {'evidence_boundary': {'facts': {'metadata.field_description': {'value': '交易日期'}}}}
        fixture = ROOT.parent / 'finance_program_skill_recovery20_20261021_r5/reviews/final-01/extract/runs/native-02-unified/call-005/returned.json'
        system = json.loads(fixture.read_bytes())['messages'][0]['content']
        self.assertTrue(system.startswith('Execute only the selected Skill INFER_CATEGORY_FROM_TOPK.'))
        messages = [{'role': 'system', 'content': system},
                    {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)}]
        actual = interface.rewrite(messages)
        self.assertIn('Field-definition classification', actual[0]['content'])
        self.assertIn('not automatically require a sampled row value', actual[0]['content'])
        self.assertEqual(json.loads(actual[-1]['content']), payload)
        self.assertNotIn('Field-definition classification', messages[0]['content'])

    def test_candidate_search_receives_same_task_definition(self):
        messages = [{'role': 'system', 'content': 'Execute only the selected Skill SELECT_CATEGORY_TOPK.'},
                    {'role': 'user', 'content': '{}'}]
        self.assertIn('Field-definition classification', interface.rewrite(messages)[0]['content'])

    def test_planner_reuses_facts_without_inserting_steps(self):
        payload = {'current': {'input': {'field_name': 'UNSEEN'}, 'named_state': {}}, 'shared_values': {},
                   'evidence_boundary': {'facts': {}}, 'remaining_budget': {'program_steps': 16}}
        messages = [{'role': 'system', 'content': 'You are the Global Program Planner.'},
                    {'role': 'user', 'content': json.dumps(payload)}]
        actual = interface.rewrite(messages)
        self.assertIn('Acquisition-aware recovery', actual[0]['content'])
        self.assertIn('Do not retry a classification after an insufficient outcome without new eligible raw facts', actual[0]['content'])
        restored = codec.expand_packet(json.loads(actual[-1]['content']))
        restored.pop('binding_status')
        self.assertEqual(restored, payload)

    def test_native_insufficiency_and_invalid_citations_still_fail(self):
        view = {'facts': {'metadata.table_name': {'value': 'SECRET', 'decisive_eligible': False}}}
        insufficient = {'option_id': 'UNRESOLVED', 'summary': 'unsupported',
                        'support': {'sufficient': False, 'citations': [], 'missing_evidence': 'unresolved subject'}}
        with self.assertRaisesRegex(ValueError, 'insufficient:'):
            boundary.classification_response(json.dumps(insufficient), view)
        insufficient['support'] = {'sufficient': True, 'citations': [{'fact_id': 'metadata.table_name', 'quote': 'SECRET'}], 'missing_evidence': ''}
        with self.assertRaisesRegex(ValueError, 'identifier_or_auxiliary:'):
            boundary.classification_response(json.dumps(insufficient), view)

    def test_original_malformed_synthesis_output_is_not_repaired(self):
        with self.assertRaises(ValueError):
            boundary.strict('{"summary":"a","data":{"metadata_semantics":"b"}')

if __name__ == '__main__':
    unittest.main()
