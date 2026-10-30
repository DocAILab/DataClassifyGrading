import importlib.util
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT.parent/'finance_program_skill_context_contract_20261021_r2'
sys.path.insert(0,str(ROOT));sys.path.insert(1,str(OLD))
p=ROOT/'single_loader.py' if (ROOT/'single_loader.py').exists() else OLD/'single_loader.py'
s=importlib.util.spec_from_file_location('single_loader_peft_regression',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class PeftMap(unittest.TestCase):
 def test_peft_never_auto_redispatches_language_or_lora_to_meta(self):
  original=(ROOT.parent/'finance_program_student_b80_20261021_r3/train_student.py').read_text(encoding='utf-8')
  source=m.adapt_source(original)
  self.assertIn('device_map=adapter_device_map()',source,'PEFT defaults auto when any base module uses CPU')
  self.assertIn('validate_adapter_placement(model)',source)
  self.assertEqual(m.adapter_device_map(),{'base_model.model.model.language_model':0,'base_model.model.lm_head':0,'base_model.model.model.visual':'cpu'})
  with self.assertRaises(ValueError):m.adapt_source(original.replace('is_trainable=True','is_trainable=False'))
if __name__=='__main__':unittest.main()
