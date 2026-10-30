import copy
import importlib
import unittest
import torch
from types import SimpleNamespace


class TinyLM(torch.nn.Module):
    def __init__(self):
        super().__init__();self.embedding=torch.nn.Embedding(13,7);self.head=torch.nn.Linear(7,13)
    def get_input_embeddings(self):return self.embedding
    def forward(self,input_ids,attention_mask=None,use_cache=False,logits_to_keep=None):
        x=self.head(self.embedding(input_ids))
        return SimpleNamespace(logits=x[:,-logits_to_keep:] if logits_to_keep else x)


class PolicyTests(unittest.TestCase):
    def mod(self):
        try:return importlib.import_module('policy_math')
        except ModuleNotFoundError:self.fail('policy_math implementation absent')

    def test_exact_completion_shift_and_eos(self):
        m=self.mod();torch.manual_seed(3);model=TinyLM();p=[1,2,3];r=[4,5,6]
        actual=m.action_log_probs(model,p,r,temperature=.8)
        logits=model(torch.tensor([p+r])).logits[0,len(p)-1:-1].float()/.8
        expected=torch.log_softmax(logits,-1).gather(1,torch.tensor(r)[:,None]).squeeze(1)
        torch.testing.assert_close(actual,expected)
        self.assertEqual(len(actual),3)

    def test_external_tokens_have_zero_gradient_and_clipping(self):
        m=self.mod();p=torch.tensor([0.,.5,8.],requires_grad=True);old=torch.zeros(3);ref=torch.zeros(3)
        loss=m.token_objective(p,old,ref,1.,torch.tensor([1.,1.,0.]),beta=0)
        loss.backward();self.assertEqual(float(p.grad[2]),0.);self.assertEqual(float(p.grad[1]),0.)
        self.assertNotEqual(float(p.grad[0]),0.)

    def test_real_torch_update_and_frozen_reference(self):
        m=self.mod();torch.manual_seed(7);model=TinyLM();reference=copy.deepcopy(model)
        for p in reference.parameters():p.requires_grad_(False)
        before={n:p.detach().clone() for n,p in model.named_parameters()};ref_before=copy.deepcopy(reference.state_dict())
        with torch.no_grad():old=m.action_log_probs(model,[1,2],[3,4]);rr=m.action_log_probs(reference,[1,2],[3,4])
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
        loss=m.token_objective(m.action_log_probs(model,[1,2],[3,4]),old,rr,1.,torch.ones(2));loss.backward();optimizer.step()
        self.assertTrue(any(not torch.equal(p,before[n]) for n,p in model.named_parameters()))
        self.assertTrue(all(torch.equal(v,ref_before[n]) for n,v in reference.state_dict().items()))
        self.assertTrue(all(p.grad is None for p in reference.parameters()))

    def test_no_prompt_truncation_or_empty_action(self):
        m=self.mod();model=TinyLM()
        for p,r in [([], [1]),([1],[]),([1]*49152,[2])]:
            with self.assertRaises(ValueError):m.action_log_probs(model,p,r)


if __name__=='__main__':unittest.main()
