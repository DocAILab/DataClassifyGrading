import inspect
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import policy_math
import execution_contract


class CorrectionTests(unittest.TestCase):
    def corrected(self):
        fn=getattr(policy_math,'corrected_token_objective',None)
        self.assertTrue(callable(fn),'rollout/training mismatch must have a separate correction, not a relaxed guard')
        return fn

    def test_same_old_actor_ratio_one_despite_sampler_numeric_difference(self):
        fn=self.corrected()
        current=torch.tensor([-1.,-2.,-3.],requires_grad=True)
        old=current.detach().clone();behavior=old+torch.tensor([.23616266,-.1,0.])
        loss=fn(current,old,behavior,old,1.,torch.ones(3),beta=0.)
        expected_weights=torch.exp(old-behavior).clamp(max=2.)
        self.assertAlmostEqual(float(loss.detach()),-float(expected_weights.mean()),places=6)
        loss.backward()
        torch.testing.assert_close(current.grad,-expected_weights/3)

    def test_importance_correction_detached_upper_capped_and_external_masked(self):
        fn=self.corrected()
        current=torch.tensor([-1.,-1.,-1.],requires_grad=True)
        old=torch.tensor([-1.,-1.,-1.],requires_grad=True)
        behavior=torch.tensor([-5.,-1.,-5.],requires_grad=True)
        fn(current,old,behavior,old.detach(),1.,torch.tensor([1.,1.,0.]),beta=0.).backward()
        torch.testing.assert_close(current.grad,torch.tensor([-1.,-.5,0.]))
        self.assertIsNone(behavior.grad);self.assertIsNone(old.grad)

    def test_clipping_uses_training_old_policy_and_keeps_kl(self):
        fn=self.corrected()
        current=torch.tensor([-.5],requires_grad=True);old=torch.tensor([-1.]);behavior=old.clone()
        fn(current,old,behavior,old,1.,torch.ones(1),beta=0.).backward()
        self.assertEqual(float(current.grad[0]),0.)
        current.grad=None
        fn(current,old,behavior,old,1.,torch.ones(1),beta=.02).backward()
        self.assertGreater(float(current.grad[0]),0.)

    def test_nonfinite_evidence_rejected(self):
        fn=self.corrected()
        with self.assertRaises((ValueError,FloatingPointError)):
            fn(torch.tensor([-1.]),torch.tensor([-1.]),torch.tensor([float('nan')]),torch.tensor([-1.]),1.,torch.ones(1))

    def test_repair_claims_carry_original_spent_budget(self):
        self.assertIn('carry_in',inspect.signature(execution_contract.Claims).parameters)
        with tempfile.TemporaryDirectory() as directory:
            c=execution_contract.Claims(Path(directory),deadline=1000,limit=566,carry_in=564)
            self.assertEqual(c.claim('actor_check_forward','turn',0,now=100),565)
            c=execution_contract.Claims(Path(directory),deadline=1000,limit=566,carry_in=564)
            self.assertEqual(c.claim('actor_gradient_forward','turn',0,now=101),566)
            with self.assertRaises(RuntimeError):c.claim('generate','new',1,now=102)


if __name__=='__main__':unittest.main()
