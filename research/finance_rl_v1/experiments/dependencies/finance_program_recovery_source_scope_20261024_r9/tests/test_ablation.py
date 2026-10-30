import importlib.util,json,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import common,interface
old=ROOT.parent/'finance_program_recovery_feedback_20261024_r5'
spec=importlib.util.spec_from_file_location('expected_verified_r5_interface',old/'interface.py')
expected=importlib.util.module_from_spec(spec);spec.loader.exec_module(expected)
sys.path.append(str(common.PARENT/'reviews'))
from message_envelope import restore

class AblationTests(unittest.TestCase):
 def test_real_closed_requests_exact_r5_without_losing_facts(self):
  fixture=ROOT.parent/'finance_program_recovery_source_scope_20261024_r8/reviews/final-01/extract/runs/017-N'
  if not fixture.exists():fixture=common.CAMPAIGN/'runs/017-N'
  actual_session=interface.RecoverySession();expected_session=expected.RecoverySession();seen=0
  for p in sorted(fixture.glob('call-*/returned.json')):
   messages=restore(json.loads(p.read_bytes())['messages'])
   actual=actual_session.rewrite(messages);wanted=expected_session.rewrite(messages)
   self.assertTrue(actual==wanted,'new interface must exactly reuse R5, not duplicate R6 source brief')
   body=json.loads(actual[-1]['content'])
   self.assertNotIn('decision_source_brief',body)
   original=json.loads(messages[-1]['content'])
   if messages[0]['content'].startswith('Execute only the selected Skill') and 'evidence_boundary' in original:
    self.assertEqual(original['evidence_boundary']['facts'],body['evidence_boundary']['facts'])
   seen+=1
  self.assertEqual(seen,13)
 def test_same_round_anchor_not_new_root(self):
  self.assertEqual(common.CAMPAIGN,ROOT.parent/'finance_program_recovery_source_scope_20261024_r8')
 def test_original_resume_function_preserves_clock_and_rejects_expired(self):
  spec=importlib.util.spec_from_file_location('old_exact_resume',old/'campaign.py')
  m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
  first=dict(gpu_started_unix=1792781732.2619646,gpu_deadline_unix=1792810532.2619646)
  result=m.resume_admission(first,dict(gpu_indices=[0,2,3]),now=1792783000,campaign_root='r8')
  self.assertEqual(result['gpu_started_unix'],first['gpu_started_unix'])
  self.assertEqual(result['gpu_deadline_unix'],first['gpu_deadline_unix'])
  with self.assertRaises(TimeoutError):m.resume_admission(first,{},now=first['gpu_deadline_unix'],campaign_root='r8')
if __name__=='__main__':unittest.main()
