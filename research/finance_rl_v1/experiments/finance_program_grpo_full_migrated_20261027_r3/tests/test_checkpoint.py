import importlib
import tempfile
import unittest
import torch
from pathlib import Path


class CheckpointTests(unittest.TestCase):
    def test_actual_weights_optimizer_rng_roundtrip_and_tamper(self):
        try:m=importlib.import_module('checkpoint_store')
        except ModuleNotFoundError:self.fail('checkpoint_store implementation absent')
        torch.manual_seed(12);p=torch.nn.Parameter(torch.tensor([1.,2.]));opt=torch.optim.AdamW([p],lr=.01)
        p.square().sum().backward();opt.step();expected=p.detach().clone();rng=torch.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d)/'v1';proof=m.save(folder,[('p',p)],opt,version=1,reference_digest='abc')
            p.data.zero_();torch.manual_seed(99)
            restored=m.restore(folder,[('p',p)],opt,reference_digest='abc')
            self.assertTrue(torch.equal(p,expected));self.assertTrue(torch.equal(torch.get_rng_state(),rng));self.assertEqual(restored['policy_version'],1)
            with (folder/'state.pt').open('ab') as h:h.write(b'tampered')
            with self.assertRaises(ValueError):m.restore(folder,[('p',p)],opt,reference_digest='abc')


if __name__=='__main__':unittest.main()
