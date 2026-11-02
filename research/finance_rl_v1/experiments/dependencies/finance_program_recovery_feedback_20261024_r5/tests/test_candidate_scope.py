import copy,json,unittest
from pathlib import Path
import interface
R4=Path(__file__).resolve().parents[2]/'finance_program_recovery_feedback_20261024_r4'
def request(slot,stage):
    root=R4/'reviews/final-01/extract' if (R4/'reviews/final-01/extract').exists() else R4
    for p in sorted((root/'runs'/slot).glob('call-*/model-returned.json')):
        value=json.loads(p.read_bytes())
        if value['stage']==stage:return json.loads((p.parent/'returned.json').read_bytes())['messages']
    raise AssertionError('real fixture missing')
class ScopeTests(unittest.TestCase):
 def test_all_selector_options_preserved_without_private_descriptions(self):
    messages=request('004-N','SELECT_CATEGORY_TOPK');before=copy.deepcopy(messages)
    body=json.loads(interface.rewrite(messages)[-1]['content']);original=json.loads(messages[-1]['content'])
    self.assertIn('candidate_scope_index',body)
    self.assertEqual(body['candidate_scope_index'],original['category_options'])
    self.assertTrue(all(set(x)=={'option_id','label'} for x in body['candidate_scope_index']))
    self.assertEqual(messages,before)
 def test_inference_scope_copies_only_supplied_definitions(self):
    messages=request('009-N','INFER_CATEGORY_FROM_TOPK');body=json.loads(interface.rewrite(messages)[-1]['content'])
    self.assertIn('candidate_scope_index',body)
    self.assertEqual(body['candidate_scope_index'],json.loads(messages[-1]['content'])['candidate_details'])
    self.assertFalse(body['candidate_scope_evidence_added'])
    self.assertIn('not ranks',interface.rewrite(messages)[0]['content'])
 def test_other_skills_do_not_receive_extra_catalog(self):
    messages=request('004-N','INTERPRET_FIELD_NAME')
    self.assertNotIn('candidate_scope_index',json.loads(interface.rewrite(messages)[-1]['content']))
if __name__=='__main__':unittest.main()
