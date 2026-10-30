import copy,json,unittest
from pathlib import Path
import interface
PARENT=Path(__file__).resolve().parents[2]/'finance_program_recovery_feedback_20261024_r5'
def request(slot,stage,*,context=False):
    root=PARENT/'reviews/final-01/extract' if (PARENT/'reviews/final-01/extract').exists() else PARENT
    for p in sorted((root/'runs'/slot).glob('call-*/model-returned.json')):
        value=json.loads(p.read_bytes())
        if value['stage']!=stage:continue
        b=json.loads(value['messages'][-1]['content'])
        if context and len([f for f in b['evidence_boundary']['facts'].values() if f.get('decisive_eligible') is True])<=2:continue
        return json.loads((p.parent/'returned.json').read_bytes())['messages']
    raise AssertionError('actual fixture missing')
class BriefTests(unittest.TestCase):
 def test_actual_same_table_facts_visible_with_exact_ids_and_values(self):
    m=request('014-N','INFER_CATEGORY_FROM_TOPK',context=True);before=copy.deepcopy(m)
    b=json.loads(m[-1]['content']);new=json.loads(interface.rewrite(m)[-1]['content'])
    self.assertIn('decision_source_brief',new)
    expected={k:f['value'] for k,f in b['evidence_boundary']['facts'].items() if f.get('decisive_eligible') is True}
    self.assertEqual({f['fact_id']:f['value'] for f in new['decision_source_brief']},expected)
    self.assertGreater(len(expected),2)
    self.assertEqual(b['evidence_boundary'],new['evidence_boundary']);self.assertEqual(m,before)
    self.assertFalse(new['source_brief_adds_evidence']);self.assertFalse(new['source_brief_is_sufficiency_judgment'])
 def test_identifiers_hypotheses_counts_excluded(self):
    m=request('001-N','INFER_CATEGORY_FROM_TOPK');b=json.loads(m[-1]['content']);new=json.loads(interface.rewrite(m)[-1]['content'])
    self.assertIn('decision_source_brief',new)
    ids={f['fact_id'] for f in new['decision_source_brief']}
    self.assertNotIn('metadata.field_name',ids)
    self.assertTrue(all(b['evidence_boundary']['facts'][i]['decisive_eligible'] is True for i in ids))
 def test_selector_keeps_its_existing_scope_not_new_fact_brief(self):
    m=request('014-N','SELECT_CATEGORY_TOPK')
    self.assertNotIn('decision_source_brief',json.loads(interface.rewrite(m)[-1]['content']))
if __name__=='__main__':unittest.main()
