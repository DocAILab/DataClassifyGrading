import importlib,unittest
class CampaignTests(unittest.TestCase):
 def test_resume_never_resets_clock_or_request_anchor(self):
    self.assertIsNotNone(importlib.util.find_spec('campaign'))
    c=importlib.import_module('campaign')
    old={'gpu_started_unix':100.,'gpu_deadline_unix':1000.,'max_sequence_tokens':49152}
    d=c.resume_admission(old,{'gpu_indices':[0,2,3]},now=200.,campaign_root='/fixed/r4')
    self.assertEqual(d['gpu_started_unix'],100.);self.assertEqual(d['gpu_deadline_unix'],1000.)
    self.assertEqual(d['campaign_root'],'/fixed/r4');self.assertEqual(old['gpu_deadline_unix'],1000.)
    with self.assertRaises(TimeoutError):c.resume_admission(old,{},now=999.,campaign_root='/fixed/r4')
if __name__=='__main__':unittest.main()
