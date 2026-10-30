import tempfile
import unittest
from pathlib import Path
import torch
import runtime_safety as safety

class SafetyTests(unittest.TestCase):
    def test_low_host_or_gpu_blocks(self):
        with self.assertRaises(RuntimeError):safety.validate_resources(63*2**30,[4*2**30]*3,30*2**30)
        with self.assertRaises(RuntimeError):safety.validate_resources(100*2**30,[2**30]*3,30*2**30)
        with self.assertRaises(RuntimeError):safety.validate_resources(100*2**30,[4*2**30]*3,2**30)
        safety.validate_resources(100*2**30,[4*2**30]*3,30*2**30)
    def test_boundary_clears_grad_and_saves_real_optimizer(self):
        with tempfile.TemporaryDirectory() as d:
            p=torch.nn.Parameter(torch.ones(3));opt=torch.optim.AdamW([p]);p.sum().backward();opt.step()
            folder=safety.persist_boundary(Path(d),[('p',p)],opt,step=1,consumed=['a'],identity={'x':1})
            self.assertIsNone(p.grad)
            self.assertTrue((folder/'manifest.json').is_file())
            state=torch.load(folder/'state.pt',weights_only=True)
            self.assertEqual(float(next(iter(state['optimizer']['state'].values()))['step']),1.)
    def test_trainer_has_real_micro_guards_and_per_step_checkpoint(self):
        import ast
        code=(Path(__file__).resolve().parents[1]/'train.py').read_text(encoding='utf-8')
        tree=ast.parse(code)
        calls=[n.func.id for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)]
        self.assertGreaterEqual(calls.count('guard_resources'),3)
        self.assertGreaterEqual(calls.count('persist_boundary'),2)
        self.assertIn('del loss',code)

if __name__=='__main__':unittest.main()
