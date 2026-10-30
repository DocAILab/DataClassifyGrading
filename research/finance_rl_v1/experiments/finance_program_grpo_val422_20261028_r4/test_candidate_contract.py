import hashlib,json,tempfile,unittest
from pathlib import Path
import torch
import candidate_contract as c

def sha_tensor(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

class CandidateContract(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
  self.weights={f'p{i}':torch.tensor([float(i)],dtype=torch.float32) for i in range(80)}
  self.state={'policy_version':5,'reference_digest':'reference','weights':self.weights}
  torch.save(self.state,self.root/'state.pt');self.sha=hashlib.sha256((self.root/'state.pt').read_bytes()).hexdigest()
  (self.root/'manifest.json').write_text(json.dumps({'state_sha256':self.sha,'reference_digest':'reference','policy_version':5,'tensor_count':80}))
 def tearDown(self):self.tmp.cleanup()
 def test_valid_checkpoint_and_real_tensor_copy(self):
  state=c.load_checkpoint(self.root,self.sha,'reference',torch)
  class B:pass
  b=B();model=torch.nn.Module()
  for i in range(80):model.register_parameter(f'p{i}',torch.nn.Parameter(torch.zeros(1)))
  b.named=list(model.named_parameters());b.runtime={'model':model};b.identity={}
  got=c.apply_checkpoint(b,state,sha_tensor,torch,self.sha)
  self.assertEqual(got,{k:sha_tensor(v) for k,v in self.weights.items()})
  self.assertTrue(all(not p.requires_grad for _,p in b.named));self.assertEqual(b.identity['active_actor_checkpoint_state_sha256'],self.sha)
 def test_mismatched_sha_or_reference_rejected(self):
  with self.assertRaises(ValueError):c.load_checkpoint(self.root,'0'*64,'reference',torch)
  with self.assertRaises(ValueError):c.load_checkpoint(self.root,self.sha,'different',torch)
 def test_wrong_tensor_names_rejected_before_copy(self):
  class B:pass
  b=B();model=torch.nn.Module()
  for i in range(80):model.register_parameter(f'p{i}',torch.nn.Parameter(torch.zeros(1)))
  b.named=list(model.named_parameters());b.runtime={'model':model};b.identity={}
  bad=dict(self.state,weights={**self.weights,'extra':torch.ones(1)})
  with self.assertRaises(ValueError):c.apply_checkpoint(b,bad,sha_tensor,torch,self.sha)
  self.assertTrue(all(torch.equal(p,torch.zeros(1)) for _,p in b.named))
if __name__=='__main__':unittest.main()
