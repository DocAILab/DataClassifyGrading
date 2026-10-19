import json
from pathlib import Path
import sys
import unittest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from program_loop import build_program_controller


class IntegrationTests(unittest.TestCase):
    def make(self, *, acquire=True, corrupt=False):
        registry = {'d:A/B/C' + str(i): {'name': 'C' + str(i), 'path': ['A', 'B', 'C' + str(i)],
                    'description': 'synthetic standard ' + str(i), 'grading_level': 'L1'} for i in range(5)}
        rows = [{'id': 'synthetic', 'domain': 'd', 'metadata': {'field_name': 'OPAQUE',
                 'field_description': '真实字段说明', 'table_name': 'TABLE', 'gold': 'SECRET_GOLD'}}]
        messages = []
        def generate(value):
            messages.append(value); head = value[0]['content'].splitlines()[0]; user = json.loads(value[-1]['content'])
            if head.startswith('You are the Global Program Planner'):
                tools = (['GET_FIELD_METADATA'] if acquire else []) + ['INTERPRET_FIELD_NAME', 'SELECT_CATEGORY_TOPK', 'INFER_CATEGORY_FROM_TOPK', 'INFER_GRADING_LEVEL', 'FINALIZE_CLASSIFICATION']
                return json.dumps({'program': [{'tool': name, 'arguments': {}} for name in tools]})
            skill = head.removeprefix('Execute only the selected Skill ').removesuffix('.')
            if skill == 'INTERPRET_FIELD_NAME':
                ctx = user.get('grounding_context', {'field_name': 'OPAQUE'})
                desc = ctx.get('field_description'); status = 'observed_description' if desc else 'unknown'
                if corrupt:desc = 'invented debt service meaning'
                return json.dumps({'summary': status, 'data': {'field_semantics': {'schema': 'grounded-source-v1',
                    'status': status, 'field_name': 'OPAQUE', 'description': desc, 'source_consistency': 'not_adjudicated'}}})
            if skill == 'SELECT_CATEGORY_TOPK':
                return json.dumps({'option_ids': [r['option_id'] for r in user['category_options'][:5]], 'summary': 'synthetic'})
            if skill == 'INFER_CATEGORY_FROM_TOPK':
                return json.dumps({'option_id': user['candidate_details'][0]['option_id'], 'summary': 'synthetic'})
            raise AssertionError('unexpected model role')
        c = build_program_controller(generate, registry, rows, {'max_programs': 1, 'max_program_steps': 10,
            'max_skill_steps': 12, 'max_planner_attempts': 1, 'max_identical_calls': 1}, skill_pool='F66')
        trace = c.run({'source_id': 'synthetic', 'domain': 'd', 'field_name': 'OPAQUE'}).to_dict()
        return trace, messages

    def test_real_controller_publishes_grounded_value_and_terminates(self):
        trace, messages = self.make()
        self.assertTrue(trace['terminated'])
        self.assertEqual(trace['state']['field_semantics']['description'], '真实字段说明')
        self.assertNotIn('SECRET_GOLD', json.dumps(messages, ensure_ascii=False))

    def test_no_hidden_retrieval_before_acquisition(self):
        trace, messages = self.make(acquire=False)
        self.assertTrue(trace['terminated'])
        call = next(json.loads(m[-1]['content']) for m in messages if 'selected Skill INTERPRET_FIELD_NAME.' in m[0]['content'])
        self.assertNotIn('field_description', call['grounding_context'])
        self.assertEqual(trace['state']['field_semantics']['status'], 'unknown')

    def test_unsupported_output_fails_without_autofinalize(self):
        trace, messages = self.make(corrupt=True)
        self.assertFalse(trace['terminated'])
        self.assertNotIn('field_semantics', trace['state'])
        self.assertTrue(any(s['status'] == 'error' for p in trace['programs'] for s in p['steps']))


if __name__ == '__main__':unittest.main()
