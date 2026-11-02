import importlib.util
from pathlib import Path
import unittest
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
OLD=ROOT.parent/'finance_program_skill_context_contract_20261021_r1'
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
P=module('single_profile_test',ROOT/'single_policy.py') if (ROOT/'single_policy.py').exists() else module('old_profile_test',OLD/'resource_profile.py')
UUID='GPU-4e00b91b-5123-d78d-35f3-e72212f3f588'
G=1024**3
def rows():return [dict(index=i,uuid=UUID if i==0 else 'GPU-other',free_bytes=23*G,total_bytes=32*G,other_compute_pids=[123],bf16_compatible=True) for i in range(2)]
class SingleGPU(unittest.TestCase):
 def test_only_gpu0_with_unchanged_two_gib_headroom(self):
  c=P.candidates(rows(),max_sequence_tokens=49152)
  self.assertEqual(len(c),1,'single GPU0 admission missing')
  self.assertEqual(c[0]['gpu_indices'],[0]);self.assertEqual(c[0]['gpu_uuids'],[UUID])
  self.assertEqual(c[0]['allocator_limit_bytes'],[20*G]);self.assertEqual(c[0]['headroom_bytes'],2*G)
  self.assertEqual(c[0]['scientific_updates_allowed'],0)
 def test_actual_identity_and_memory_are_rechecked(self):
  c=P.candidates(rows(),max_sequence_tokens=49152)
  self.assertTrue(c,'no single-card resource path')
  for key,value in [('free_bytes',22*G-1),('uuid','GPU-other')]:
   r=rows();r[0][key]=value
   with self.assertRaises(ValueError):P.recheck(r,c[0])
 def test_inference_admission_is_one_gpu_and_original_deadline(self):
  fn=getattr(P,'validate_admission',None);self.assertIsNotNone(fn,'single GPU admission missing')
  a=dict(gpu_indices=[0],gpu_uuids=[UUID],allocator_limit_bytes=[20*G],model_max_memory_bytes=[18*G],gpu_deadline_unix=1792587659.6128213,context_window_tokens=49152,max_sequence_tokens=49152)
  fn(a,now=1792583600)
  for patch in [dict(gpu_uuids=[UUID,'GPU-other']),dict(gpu_deadline_unix=1792587660),dict(allocator_limit_bytes=[21*G])]:
   with self.assertRaises(ValueError):fn(dict(a,**patch),now=1792583600)
 def test_only_unused_vision_is_on_cpu_and_unknown_loader_refused(self):
  p=ROOT/'single_loader.py';self.assertTrue(p.exists(),'single GPU loader missing')
  m=module('single_loader_test',p)
  self.assertEqual(m.device_map(),{'model.language_model':0,'lm_head':0,'model.visual':'cpu'})
  p=ROOT.parent/'finance_program_student_b80_20261021_r3/train_student.py'
  changed=m.adapt_source(p.read_text(encoding='utf-8'))
  self.assertIn('validate_placement(base.hf_device_map)',changed)
  self.assertIn('device_map=device_map()',changed)
  with self.assertRaises(ValueError):m.adapt_source('unknown changed loader')
if __name__=='__main__':unittest.main()
