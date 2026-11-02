import unittest,importlib.util,tempfile,json
from pathlib import Path
class RecoveryTests(unittest.TestCase):
 def api(self,n):
  p=Path(__file__).resolve().parents[1]/(n+'.py');self.assertTrue(p.exists(),n+' missing');s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
 def test_dynamic_three_cards_use_free_memory_and_exact_uuid(self):
  m=self.api('resource_runtime');p={'gpu_cap_GiB':26,'gpu_reserve_GiB':2,'gpu_admission_free_GiB':29,'allowed_gpu_uuids':{'0':'GPU-a','4':'GPU-b','6':'GPU-c','7':'GPU-d'}}
  rows=[dict(index=i,uuid=u,free_bytes=int(f*1024**3),bf16_compatible=True,other_compute_pids=[]) for i,u,f in [(0,'GPU-a',15),(4,'GPU-b',29.79),(6,'GPU-c',29.79),(7,'GPU-d',29.79)]]
  c=m.candidates(rows,p);self.assertEqual(c[0]['gpu_indices'],[4,6,7]);self.assertEqual(c[0]['headroom_bytes'],2*1024**3)
 def test_unknown_uuid_not_used(self):
  m=self.api('resource_runtime');p={'gpu_cap_GiB':26,'gpu_reserve_GiB':2,'gpu_admission_free_GiB':29,'allowed_gpu_uuids':{}}
  self.assertEqual(m.candidates([dict(index=4,uuid='GPU-unknown',free_bytes=32*1024**3,bf16_compatible=True,other_compute_pids=[])],p),[])
 def test_completed_episode_is_reused_without_relocating(self):
  m=self.api('recovery');
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'episode';p.mkdir();(p/'result.json').write_text('{}');self.assertFalse(m.preserve_incomplete(p,Path(d)/'old',1));self.assertTrue((p/'result.json').exists())
 def test_partial_episode_is_preserved_not_deleted(self):
  m=self.api('recovery');
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'episode';p.mkdir();(p/'evidence.json').write_text('original');dest=m.preserve_incomplete(p,Path(d)/'old',2);self.assertFalse(p.exists());self.assertEqual((dest/'evidence.json').read_text(),'original')
 def test_completed_episode_sha_tamper_refused(self):
  m=self.api('recovery')
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'r.json';p.write_text('one')
   with self.assertRaises(ValueError):m.require_hash(p,'bad')
 def test_live_attempt_never_relaunched(self):self.assertEqual(self.api('recovery').next_action(running=True,exit_code=None,completed=False,recoverable=True,attempts=2,limit=8),'WAIT')
 def test_numerical_failure_needs_diagnosis(self):self.assertEqual(self.api('recovery').next_action(running=False,exit_code=1,completed=False,recoverable=False,attempts=2,limit=8),'DIAGNOSE')
if __name__=='__main__':unittest.main()
