import unittest,importlib.util
from pathlib import Path
class FullContract(unittest.TestCase):
 def api(self,name):
  p=Path(__file__).resolve().parents[1]/(name+'.py');self.assertTrue(p.exists(),name+' absent')
  s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
 def rows(self,version=0):
  return [dict(source_id=str(s),sample_index=i,policy_version=version,reward={'A1':float(s==0 and i<2),'outcome':int(s==0 and i<2)},legal_completion=True,native_replay_exact=True,actual_initial_exact=True) for s in range(8) for i in range(4)]
 def test_actual_seventh_optimizer_refused_before_forward(self):
  import policy_update
  class B:
   admission={'maximum_optimizer_updates':6}
   version=6
  with self.assertRaises(ValueError):policy_update.update(B(),[],'A1')
 def test_all48_fixed_unique_six_batches(self):
  m=self.api('full_contract');s=m.schedule([str(i) for i in range(48)],[]);self.assertEqual(len(s),6);self.assertEqual(len(set(sum(s,[]))),48)
 def test_eval_overlap_refused(self):
  with self.assertRaises(ValueError):self.api('full_contract').schedule([str(i) for i in range(48)],['2'])
 def test_single_mixed_group_is_real_signal(self):self.assertTrue(self.api('full_contract').gate(self.rows(),0)['allow_update'])
 def test_all_equal_groups_have_no_fake_advantage(self):
  a=self.rows()
  for v in a:v['reward']={'A1':0.,'outcome':0}
  self.assertFalse(self.api('full_contract').gate(a,0)['allow_update'])
 def test_stale_policy_refused(self):
  a=self.rows();a[-1]['policy_version']=1
  with self.assertRaises(ValueError):self.api('full_contract').groups(a,0)
 def test_partial_groups_refused(self):
  with self.assertRaises(ValueError):self.api('full_contract').groups(self.rows()[:-1],0)
 def test_budget_deducts_every_attempt_and_refuses_reset(self):
  m=self.api('campaign_budget');p={'training_seconds':100,'absolute_end_unix':500,'training_calls':1000,'maximum_attempts':3}
  a=m.remaining(p,[{'gpu_seconds':20,'claims':50}],now=200);self.assertEqual(a['seconds'],80);self.assertEqual(a['calls'],950)
 def test_budget_refuses_fourth_attempt(self):
  with self.assertRaises(ValueError):self.api('campaign_budget').remaining({'training_seconds':100,'absolute_end_unix':500,'training_calls':1000,'maximum_attempts':3},[{'gpu_seconds':1,'claims':1}]*3,now=200)
if __name__=='__main__':unittest.main()
