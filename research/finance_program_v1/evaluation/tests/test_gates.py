import unittest
import tempfile
from pathlib import Path
import hashlib
try:
    from gate import training_ready, enabled_for, validate_ids, eos_options
except ImportError:
    training_ready = enabled_for = validate_ids = None


class GateTests(unittest.TestCase):
    def test_changed_adapter_after_training_acceptance_rejected(self):
        import gate
        check=getattr(gate,'validate_adapter_files',None)
        self.assertIsNotNone(check,'missing final adapter load-time SHA gate')
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'adapter_model.safetensors').write_bytes(b'first');(root/'adapter_config.json').write_bytes(b'{}')
            anchor={'adapter_sha256':hashlib.sha256(b'first').hexdigest(),'adapter_config_sha256':hashlib.sha256(b'{}').hexdigest()}
            check(root,anchor)
            (root/'adapter_model.safetensors').write_bytes(b'changed')
            with self.assertRaises(ValueError):check(root,anchor)
    def test_ready_requires_all_actual_exits(self):
        self.assertIsNotNone(training_ready)
        state={'manifest_sha256':'sig','writer_live':False,'exits':{k:0 for k in ('trial','formal','pipeline','archive')},
            'completion':{'global_step':13,'sources':200,'epoch':1.,'manifest_sha256':'sig','adapter_sha256':'a'},
            'cpu_proof':{'independent_cpu_audit_passed':True,'manifest_sha256':'sig','adapter_sha256':'a'},'adapter_sha256':'a'}
        self.assertTrue(training_ready(state,'sig'))
        state['exits']['formal']=None
        self.assertFalse(training_ready(state,'sig'))

    def test_wrong_signature_or_live_writer_rejected(self):
        self.assertIsNotNone(training_ready)
        self.assertFalse(training_ready({'manifest_sha256':'wrong','writer_live':True},'sig'))

    def test_eos_covers_actual_base_and_supervised_eos(self):
        self.assertEqual(eos_options(248044,248046),[248044,248046])

    def test_trusted_planner_only(self):
        self.assertIsNotNone(enabled_for)
        system='trusted planner'
        self.assertTrue(enabled_for('sft',[{'role':'system','content':system}],system))
        self.assertFalse(enabled_for('base',[{'role':'system','content':system}],system))
        self.assertFalse(enabled_for('sft',[{'role':'user','content':system}],system))
        self.assertFalse(enabled_for('sft',[{'role':'system','content':'skill'},{'role':'user','content':system}],system))

    def test_ids_full_fixed_denominator_and_disjoint(self):
        self.assertIsNotNone(validate_ids)
        ids=[str(i) for i in range(80)]
        validate_ids(ids,['smoke-a','smoke-b'],{'train'})
        with self.assertRaises(ValueError):validate_ids(ids[:-1],['smoke-a','smoke-b'],set())
        with self.assertRaises(ValueError):validate_ids(ids,['0','smoke-b'],set())
        with self.assertRaises(ValueError):validate_ids(ids,['smoke-a','smoke-b'],{'0'})


if __name__=='__main__':unittest.main()
