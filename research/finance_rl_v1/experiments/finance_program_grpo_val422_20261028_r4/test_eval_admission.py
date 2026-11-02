import ast,unittest
from pathlib import Path
class AdmissionContract(unittest.TestCase):
 def test_evaluation_keeps_identity_checks_but_forbids_optimizer(self):
  source=(Path('../finance_program_termination_first_20261025_r2/live_backend.py')).read_text(encoding='utf-8')
  original="('base_model_dir','max_sequence_tokens','scientific_updates_allowed')"
  self.assertEqual(source.count(original),1)
  transformed=source.replace(original,"('base_model_dir','max_sequence_tokens')")
  self.assertIn("for key in ('base_model_dir','max_sequence_tokens')",transformed)
  wrapper=Path('live_backend.py').read_text(encoding='utf-8')
  self.assertIn("admission.get('scientific_updates_allowed')!=0",wrapper)
  self.assertIn("admission.get('maximum_optimizer_updates')!=0",wrapper)
 def test_preflight_uses_zero_update_evaluation_admission(self):
  tree=ast.parse(Path('preflight.py').read_text(encoding='utf-8'))
  calls=[x for x in ast.walk(tree) if isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute) and x.func.attr=='update' and isinstance(x.func.value,ast.Name) and x.func.value.id=='admission']
  self.assertEqual(len(calls),1);kwargs={k.arg:ast.literal_eval(k.value) for k in calls[0].keywords if k.arg in ('scientific_updates_allowed','maximum_optimizer_updates')}
  self.assertEqual(kwargs,{'scientific_updates_allowed':0,'maximum_optimizer_updates':0})
if __name__=='__main__':unittest.main()
