import importlib.util
from pathlib import Path
import unittest
ROOT=Path(__file__).resolve().parents[1]
old=ROOT.parent/'finance_program_skill_context_contract_20261021_r1/resource_profile.py'
p=ROOT/'resource_profile.py' if (ROOT/'resource_profile.py').exists() else old
s=importlib.util.spec_from_file_location('two_card_inference_profile_test',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
G=1024**3
def rows():return [dict(index=i,uuid='GPU-'+str(i),free_bytes=19*G,total_bytes=32*G,other_compute_pids=[123],bf16_compatible=True) for i in range(2)]
class TwoGPU(unittest.TestCase):
 def test_two_inference_cards_with_bounded_workspace_and_no_updates(self):
  c=m.candidates(rows(),max_sequence_tokens=49152)
  self.assertEqual(len(c),1,'inference-only two-card envelope missing')
  self.assertEqual(c[0]['allocator_limit_bytes'],[16*G]*2)
  self.assertEqual(c[0]['model_max_memory_bytes'],[12*G]*2,'10GiB native balanced map offloads lm_head to disk; CPU meta reproduction preserved')
  self.assertEqual(c[0]['headroom_bytes'],2*G)
  self.assertEqual(c[0]['scientific_updates_allowed'],0)
 def test_locked_identity_and_headroom_cannot_be_waived(self):
  c=m.candidates(rows(),max_sequence_tokens=49152);self.assertTrue(c)
  for key,value in [('uuid','GPU-replaced'),('free_bytes',18*G-1)]:
   r=rows();r[0][key]=value
   with self.assertRaises(ValueError):m.recheck(r,c[0])
  with self.assertRaises(ValueError):m.candidates(rows(),max_sequence_tokens=49153)
if __name__=='__main__':unittest.main()
