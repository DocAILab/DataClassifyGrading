import importlib
import tempfile
import unittest
from pathlib import Path


class ExecutionTests(unittest.TestCase):
    def mod(self):
        try:return importlib.import_module('execution_contract')
        except ModuleNotFoundError:self.fail('execution_contract implementation absent')

    def test_exact_bounded_planner_sampling_and_greedy_skills(self):
        m=self.mod();p=m.options('Planner',1000,now=100)
        self.assertTrue(p['do_sample']);self.assertEqual(p['temperature'],.8);self.assertEqual(p['top_k'],0);self.assertEqual(p['top_p'],1.)
        self.assertFalse(m.options('SELECT_CATEGORY_TOPK',1000,now=100)['do_sample'])
        self.assertEqual(p['max_new_tokens'],512);self.assertEqual(p['eos_token_id'],[248044,248046])
        with self.assertRaises(TimeoutError):m.options('Planner',100,now=99)

    def test_claims_monotone_and_absolute_budget_not_reset(self):
        m=self.mod()
        with tempfile.TemporaryDirectory() as directory:
            c=m.Claims(Path(directory),deadline=1000,limit=2)
            self.assertEqual(c.claim('generate','one',0,now=100),1)
            c2=m.Claims(Path(directory),deadline=1000,limit=2)
            self.assertEqual(c2.claim('ref_forward','two',0,now=101),2)
            with self.assertRaises(RuntimeError):c2.claim('generate','three',1,now=102)
            with self.assertRaises(TimeoutError):c2.claim('generate','three',1,now=1000)

    def test_bad_group_incomplete_or_cross_version_rejected(self):
        m=self.mod();rows=[dict(source_id='x',sample_index=i,policy_version=0) for i in range(4)]
        m.validate_group(rows)
        with self.assertRaises(ValueError):m.validate_group(rows[:3])
        rows[-1]['policy_version']=1
        with self.assertRaises(ValueError):m.validate_group(rows)


if __name__=='__main__':unittest.main()
