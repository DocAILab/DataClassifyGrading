import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]

class NativeSkillTest(unittest.TestCase):
    def test_original_failure_rejected_and_complete_native_shape_accepted(self):
        spec = importlib.util.spec_from_file_location('target_native_init', ROOT / 'native_skill.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        parse, skill = module.native_skill()
        old = ROOT.parent / 'finance_program_skill_recovery20_20261021_r5/reviews/final-01/extract/runs/native-17-unified/call-009/model-returned.json'
        with self.assertRaisesRegex(Exception, 'invalid JSON'):
            parse(json.loads(old.read_bytes())['raw_text'], skill)
        self.assertEqual(parse('{"summary":"observed","data":{"metadata_semantics":"Observed: provided description. Unresolved: absent values."}}', skill).status, 'success')

if __name__ == '__main__':
    unittest.main()
