import json
from pathlib import Path
import tempfile
import unittest
import torch
from recovery_safety import save_checkpoint, load_checkpoint, require_host_headroom, HostMemoryWait

class RecoverySafetyTests(unittest.TestCase):
    def make(self):
        p=torch.nn.Parameter(torch.tensor([1.,2.],dtype=torch.float32))
        return p,torch.optim.AdamW([p],lr=.0001)

    def step(self,p,opt):
        opt.zero_grad(set_to_none=True)
        (p*torch.rand_like(p)).sum().backward();opt.step();opt.zero_grad(set_to_none=True)

    def test_model_optimizer_rng_resume_matches_uninterrupted(self):
        torch.manual_seed(17);p,opt=self.make();self.step(p,opt)
        with tempfile.TemporaryDirectory() as folder:
            saved=save_checkpoint(folder,[('p',p)],opt,step=1,consumed=['s1'],identity={'data':'sha'})
            self.step(p,opt);expected=p.detach().clone()
            q,other=self.make();meta=load_checkpoint(saved,[('p',q)],other,identity={'data':'sha'})
            self.assertEqual(meta['consumed'],['s1']);self.step(q,other)
            torch.testing.assert_close(expected,q,rtol=0,atol=0)
            self.assertEqual(other.state[q]['step'].item(),2)

    def test_changed_data_is_rejected(self):
        p,opt=self.make()
        with tempfile.TemporaryDirectory() as folder:
            saved=save_checkpoint(folder,[('p',p)],opt,step=0,consumed=[],identity={'data':'sha'})
            with self.assertRaisesRegex(ValueError,'identity'):
                load_checkpoint(saved,[('p',p)],opt,identity={'data':'other'})

    def test_corrupt_bytes_rejected_before_deserialization(self):
        p,opt=self.make()
        with tempfile.TemporaryDirectory() as folder:
            saved=save_checkpoint(folder,[('p',p)],opt,step=0,consumed=[],identity={})
            with (saved/'state.pt').open('ab') as f:f.write(b'corrupt')
            with self.assertRaisesRegex(ValueError,'SHA'):
                load_checkpoint(saved,[('p',p)],opt,identity={})

    def test_existing_checkpoint_never_overwritten(self):
        p,opt=self.make()
        with tempfile.TemporaryDirectory() as folder:
            save_checkpoint(folder,[('p',p)],opt,step=0,consumed=[],identity={})
            with self.assertRaises(FileExistsError):
                save_checkpoint(folder,[('p',p)],opt,step=0,consumed=[],identity={})

    def test_partial_bundle_is_not_loadable(self):
        p,opt=self.make()
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FileNotFoundError):
                load_checkpoint(Path(folder),[('p',p)],opt,identity={})

    def test_accumulated_gradients_cannot_be_silently_dropped(self):
        p,opt=self.make();p.sum().backward()
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError,'boundary'):
                save_checkpoint(folder,[('p',p)],opt,step=0,consumed=[],identity={})

    def test_low_host_memory_defers_instead_of_training(self):
        with self.assertRaises(HostMemoryWait):require_host_headroom(available_bytes=31,required_bytes=32)
        require_host_headroom(available_bytes=32,required_bytes=32)

if __name__=='__main__':unittest.main()
