import copy,json,unittest
from pathlib import Path
import resume_contract as resume

class ResumeContractTests(unittest.TestCase):
    def fixture(self):
        parent=Path(__file__).resolve().parents[2]/'finance_program_student_fresh162_20261022_r3'
        saved=parent/'runs/training/checkpoints/checkpoint-08/manifest.json'
        if not saved.exists():saved=parent/'reviews/checkpoint-backups/checkpoint-08/manifest.json'
        meta=json.loads(saved.read_bytes())
        rows=[json.loads(x) for x in (parent/'data/train162.jsonl').read_bytes().splitlines()]
        admission=copy.deepcopy(meta['identity']['admission'])
        return meta,rows,meta['identity']['freeze_sha256'],meta['identity']['train_sha256'],admission
    def validator(self):
        self.assertTrue(callable(getattr(resume,'validate_resume',None)),'missing resume identity validator')
        return resume.validate_resume
    def test_exact128_prefix_is_only_boundary(self):
        args=self.fixture();self.assertEqual(self.validator()(*args,now=args[-1]['gpu_started_unix']+10),128)
    def test_wrong_prefix_is_rejected(self):
        args=list(self.fixture());args[0]['consumed'][0]='forged'
        with self.assertRaisesRegex(ValueError,'prefix'):self.validator()(*args,now=args[-1]['gpu_started_unix']+10)
    def test_new_window_is_rejected(self):
        args=list(self.fixture());args[-1]['gpu_deadline_unix']+=28800
        with self.assertRaisesRegex(ValueError,'clock'):self.validator()(*args,now=args[-1]['gpu_started_unix']+10)
    def test_source_identity_mismatch_rejected(self):
        args=list(self.fixture());args[2]='bad'
        with self.assertRaisesRegex(ValueError,'identity'):self.validator()(*args,now=args[-1]['gpu_started_unix']+10)
    def test_only34_samples_in_three_updates(self):
        self.assertTrue(callable(getattr(resume,'remaining_batches',None)),'missing remaining batch scheduler')
        self.assertEqual(resume.remaining_batches(list(range(162)),8),[(9,list(range(128,144))),(10,list(range(144,160))),(11,[160,161])])
    def test_any_three_bf16_gpus_without_lowering_headroom(self):
        import resource_profile
        rows=[dict(index=i,uuid='GPU-'+str(i),bf16_compatible=True,free_bytes=31*2**30,other_compute_pids=[]) for i in (5,6,7)]
        found=resource_profile.candidates(rows)
        self.assertEqual(len(found),1,'other BF16 cards must be permitted')
        self.assertEqual(found[0]['gpu_indices'],[5,6,7]);self.assertEqual(found[0]['headroom_bytes'],2*2**30)
        rows[0]['free_bytes']=29*2**30;self.assertEqual(resource_profile.candidates(rows),[])

if __name__=='__main__':unittest.main()
