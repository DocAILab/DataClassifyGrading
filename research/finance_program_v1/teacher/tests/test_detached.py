import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class DetachedTests(unittest.TestCase):
    def load(self):
        spec = importlib.util.spec_from_file_location('detached_under_test', ROOT/'detached.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_select_ids_excludes_all_used_and_never_overfills(self):
        m = self.load()
        self.assertEqual(m.select_ids([{'id':str(i)} for i in range(8)], {'0','1'}, 3), ['2','3','4'])

    def test_duplicate_source_is_rejected(self):
        m = self.load()
        with self.assertRaises(ValueError):
            m.select_ids([{'id':'x'},{'id':'x'}], set(), 20)

    def test_verify_detects_modified_file(self):
        m = self.load()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'a').write_text('before')
            manifest = {'files':[{'path':'a','sha256':m.sha(root/'a')}]}
            m.verify_files(root, manifest)
            (root/'a').write_text('after')
            with self.assertRaises(AssertionError):m.verify_files(root, manifest)

    def test_seal_verifies_archive_and_refuses_overwrite(self):
        m = self.load()
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/'trace.json').write_text('{}')
            receipt=m.seal(root)
            self.assertEqual(receipt['members'],1)
            self.assertEqual(receipt['sha256'],m.sha(root/'evidence.tar.gz'))
            with self.assertRaises(FileExistsError):m.seal(root)

    def test_inherited_budget_does_not_reset(self):
        m = self.load()
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            events=[{'type':'limits','request_cap':1200,'known_token_cap':10000000,'failed_attempt_cap':8}]
            for i in range(1,1200):
                events.extend([{'type':'claim','id':i,'pid':1},{'type':'result','id':i,'complete':True,'known_total_tokens':1,'unknown_usage':False}])
            (root/'events.jsonl').write_text('\n'.join(json.dumps(x) for x in events)+'\n')
            budget=m.PersistentBudget(root)
            self.assertEqual(budget.snapshot()['requests'],1199)
            budget.claim();budget.record({'complete':True,'usage':{'total_tokens':1}})
            with self.assertRaises(RuntimeError):budget.claim()
            self.assertEqual(budget.snapshot()['requests'],1200)


if __name__=='__main__':unittest.main()
