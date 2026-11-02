import json,unittest
from pathlib import Path
from full_policy import no_labels

class CohortContract(unittest.TestCase):
 def test_original_422_unique_and_only_n_arm(self):
  slots=json.loads(Path('selection.json').read_bytes())['slots']
  cases=json.loads(Path('eval-inputs.json').read_bytes())['cases']
  self.assertEqual(len(slots),422);self.assertEqual(len({x['source_id'] for x in slots}),422)
  self.assertEqual({x['source_id'] for x in slots},{x['source_id'] for x in cases})
  self.assertTrue(all(x['arm']=='N' for x in slots))
 def test_public_metadata_contains_no_labels(self):
  data=json.loads(Path('eval-inputs.json').read_bytes());scopes=json.loads(Path('source-scopes.json').read_bytes())
  no_labels(data['study']['rows']);no_labels(scopes['rows'])
 def test_label_in_public_metadata_is_rejected(self):
  with self.assertRaises(ValueError):no_labels([{'metadata':{'grading_level':3}}])
if __name__=='__main__':unittest.main()
