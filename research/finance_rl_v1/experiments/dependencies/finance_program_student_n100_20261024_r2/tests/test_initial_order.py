import importlib.util,json,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class InitialOrderTests(unittest.TestCase):
 def test_original_initial_state_serialization_exact_not_only_json_equal(self):
  if (ROOT/'initial_identity.py').exists():
   spec=importlib.util.spec_from_file_location('n100_initial_identity_test',ROOT/'initial_identity.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);data=m.load_inputs(ROOT/'eval-inputs.json')
  else:data=json.loads((ROOT/'eval-inputs.json').read_bytes())
  old=json.loads((ROOT.parent/'finance_program_student_fresh162_20261022_r1/eval-inputs.json').read_bytes());prior={c['source_id']:c for c in old['cases']}
  matched=0
  for c in data['cases']:
   if c['source_id'] not in prior:continue
   self.assertEqual(c,prior[c['source_id']])
   self.assertEqual(json.dumps(c,ensure_ascii=False),json.dumps(prior[c['source_id']],ensure_ascii=False),'initial prompt embeds state as text; preserve original key order')
   matched+=1
  self.assertEqual(matched,40)
