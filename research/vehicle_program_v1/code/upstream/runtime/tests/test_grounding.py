import copy
from pathlib import Path
import sys
import unittest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'vendor'))
import grounded_semantics as g


class GroundingTests(unittest.TestCase):
    def context(self):return {'field_name': 'OPAQUE', 'field_description': '真实字段说明'}
    def value(self):
        return {'schema': 'grounded-source-v1', 'status': 'observed_description', 'field_name': 'OPAQUE',
                'description': '真实字段说明', 'source_consistency': 'not_adjudicated'}

    def test_exact_observed_description_passes_without_mutation(self):
        value = self.value(); before = copy.deepcopy(value)
        self.assertTrue(g.validate_value(value, self.context()))
        self.assertEqual(value, before)

    def test_unknown_only_when_no_observed_description(self):
        value = self.value(); value.update(status='unknown', description=None)
        self.assertTrue(g.validate_value(value, {'field_name': 'OPAQUE'}))
        with self.assertRaises(ValueError):g.validate_value(value, self.context())

    def test_fabricated_or_extra_claims_rejected(self):
        for change in ({'description': 'invented business meaning'}, {'guess': 'vaccine'},
                       {'source_consistency': 'verified'}, {'status': 'conflict'}, {'field_name': 'OTHER'}):
            value = self.value(); value.update(change)
            with self.assertRaises(ValueError):g.validate_value(value, self.context())

    def test_unobtained_context_cannot_support_description(self):
        with self.assertRaises(ValueError):g.validate_value(self.value(), {'field_name': 'OPAQUE'})

    def test_string_or_incomplete_object_rejected(self):
        for value in ('guess', {}, None, {'status': 'unknown'}):
            with self.assertRaises(ValueError):g.validate_value(value, self.context())


if __name__ == '__main__':unittest.main()
