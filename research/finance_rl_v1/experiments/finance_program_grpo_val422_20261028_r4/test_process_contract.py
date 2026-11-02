import unittest
from process_contract import is_training_writer
class ProcessContract(unittest.TestCase):
 def test_shared_python_interpreter_is_not_a_training_writer(self):
  p='/home/zhubh/finance_program_grpo_full_migrated_20261027_r3';e='/home/zhubh/finance_program_grpo_val422_20261028_r2'
  self.assertFalse(is_training_writer((p+'/env/bin/python -B '+e+'/supervisor.py').encode(),'S',p,e))
  self.assertTrue(is_training_writer((p+'/env/bin/python -B '+p+'/runner.py').encode(),'S',p,e))
  self.assertFalse(is_training_writer((p+'/env/bin/python -B '+p+'/runner.py').encode(),'Z',p,e))
if __name__=='__main__':unittest.main()
