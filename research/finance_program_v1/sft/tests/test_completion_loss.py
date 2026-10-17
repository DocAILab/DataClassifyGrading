import unittest
try:
    import torch
    from completion_loss import completion_cross_entropy
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'actual torch CPU loss checks run in existing remote environment')
class CompletionLossTests(unittest.TestCase):
    def test_suffix_matches_full_causal_loss_and_gradient(self):
        torch.manual_seed(17)
        logits = torch.randn(1, 11, 29, requires_grad=True)
        labels = torch.tensor([[-100] * 7 + [2, 7, 4, 1]])
        full = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 29), labels[:, 1:].reshape(-1))
        suffix = completion_cross_entropy(logits[:, 6:], labels)
        self.assertTrue(torch.allclose(full, suffix))
        g1 = torch.autograd.grad(full, logits, retain_graph=True)[0]
        g2 = torch.autograd.grad(suffix, logits)[0]
        self.assertTrue(torch.allclose(g1, g2))

    def test_mask_hole_rejected(self):
        labels = torch.tensor([[-100, 1, -100, 2]])
        with self.assertRaises(ValueError):
            completion_cross_entropy(torch.randn(1, 3, 9), labels)

    def test_missing_boundary_logits_rejected(self):
        with self.assertRaises(ValueError):
            completion_cross_entropy(torch.randn(1, 2, 9), torch.tensor([[-100, 1, 2]]))
