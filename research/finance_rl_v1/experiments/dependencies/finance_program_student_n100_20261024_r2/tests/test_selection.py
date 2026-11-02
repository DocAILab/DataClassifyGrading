import importlib.util,json,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parent
def active():
 if not (ROOT/'selection_policy.py').exists():return None
 s=importlib.util.spec_from_file_location('n100_selection_test',ROOT/'selection_policy.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
class SelectionTests(unittest.TestCase):
 def test_single_arm100_not_legacy_dual80(self):
  m=active();r=m.build() if m else json.loads((BASE/'finance_program_student_fresh162_20261022_r1/selection.json').read_bytes())
  self.assertEqual(len(r['slots']),100,'new request is100 student-only, not legacy F/N80')
  self.assertTrue(all(s['arm']=='N' for s in r['slots']))
 def test_real_46_54_unique_source_group_strata(self):
  m=active();self.assertIsNotNone(m,'label-free explicit stratification required')
  r=m.build();slots=r['slots'];train=json.loads((BASE/'finance_program_student_fresh162_20261022_r1/data/data-lock.json').read_bytes())['identities']
  ts={x['source_id'] for x in train};tg={x['group'] for x in train}
  self.assertEqual(len({x['source_id'] for x in slots}),100);self.assertEqual(len({x['group'] for x in slots}),100)
  unseen=[x for x in slots if x['stratum']=='train_isolated'];seen=[x for x in slots if x['stratum']=='train_seen']
  self.assertEqual(len(unseen),46);self.assertEqual(len(seen),54)
  self.assertTrue(all(x['source_id'] not in ts and x['group'] not in tg for x in unseen))
  self.assertTrue(all(x['source_id'] in ts and x['group'] in tg for x in seen))
  self.assertFalse(r['labels_used']);self.assertEqual(r,m.build())
 def test_exact_eight_reuse_in_original_positions(self):
  m=active();self.assertIsNotNone(m,'explicit immutable reuse identity required')
  r=m.build();old=json.loads((BASE/'finance_program_evidence_consumption_20261024_r11/reviews/targeted-01/extract/runs/targeted-completion.json').read_bytes())['rows']
  by={s['slot_id']:s for s in r['slots']}
  self.assertEqual(len(old),8)
  for x in old:
   self.assertEqual(x['source_id'],by[x['slot_id']]['source_id']);self.assertEqual(x['group'],by[x['slot_id']]['group'])
if __name__=='__main__':unittest.main()
