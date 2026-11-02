import json,tempfile,unittest
from pathlib import Path
from migration_contract import publish_configuration,require_origin_pause,logical_destination,policy_for_attempt

class MigrationResumeTests(unittest.TestCase):
    def test_reuses_equal_checkpoint_configuration_without_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'config.json';p.write_text('{"a":1}');before=p.read_bytes()
            self.assertFalse(publish_configuration(p,{'a':1}))
            self.assertEqual(before,p.read_bytes())

    def test_configuration_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'config.json';p.write_text('{"a":1}')
            with self.assertRaises(ValueError):publish_configuration(p,{'a':2})

    def test_first_configuration_is_exclusive(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'config.json';self.assertTrue(publish_configuration(p,{'a':1}))
            self.assertEqual(json.loads(p.read_bytes()),{'a':1})

    def test_explicit_pause_is_accepted_with_original_nonzero_exits(self):
        require_origin_pause({'paused_by_user':True,'completed':False,'all_training_processes_stopped':True,'episodes':96,'actual_complete_updates':2,'third_optimizer_started':False,'original_worker_exit':{'actual_returncode':-2},'original_launch_exit':{'actual_returncode':1},'original_failure':{'type':'KeyboardInterrupt'}})

    def test_fake_success_cannot_be_used_as_pause_evidence(self):
        with self.assertRaises(ValueError):require_origin_pause({'paused_by_user':True,'original_worker_exit':{'actual_returncode':0}})

    def test_destination_keeps_exact_host_lineage(self):
        roots=['/old/r2','/new/r3']
        self.assertTrue(logical_destination('/old/r2/runs/episodes/b1/call-001','runs/episodes/b1/call-001',roots))
        self.assertTrue(logical_destination('/new/r3/runs/episodes/b1/call-001','runs/episodes/b1/call-001',roots))
        self.assertFalse(logical_destination('/unrelated/runs/episodes/b1/call-001','runs/episodes/b1/call-001',roots))

    def test_historical_gpu_policy_is_not_rewritten(self):
        old={'allowed_gpu_uuids':{'4':'old'}};new={'allowed_gpu_uuids':{'0':'new'}}
        self.assertEqual(policy_for_attempt(1,old,new),old)
        self.assertEqual(policy_for_attempt(2,old,new),new)

if __name__=='__main__':unittest.main()
