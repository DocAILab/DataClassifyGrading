"""CPU regression: sharded replay and uninterrupted accumulation make the same AdamW step."""
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import torch
from campaign import lock,write
import grpo
import parallel_replay


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input=torch.nn.Embedding(8,8)
        self.layers=torch.nn.Module()
        self.layers.lora_q=torch.nn.ModuleDict({"planner":torch.nn.Linear(8,8,bias=False)})
        self.reference=deepcopy(self.layers.lora_q["planner"])
        self.actor=True

    def get_input_embeddings(self):
        return self.input

    def forward(self,input_ids,logits_to_keep,**kwargs):
        layer=self.layers.lora_q["planner"] if self.actor else self.reference
        return SimpleNamespace(logits=layer(self.input(input_ids)[:,-logits_to_keep:]))


class Backend:
    def __init__(self,model):
        self.model=model
    def planner_parameters(self):
        return list(self.model.layers.lora_q["planner"].parameters())
    @contextmanager
    def route(self,name,learn=False):
        self.model.actor=name=="planner"
        yield


def main():
    torch.manual_seed(6)
    first=Backend(Tiny())
    second=Backend(deepcopy(first.model))
    opt1=torch.optim.AdamW(first.planner_parameters(),lr=5e-6,weight_decay=0.)
    opt2=torch.optim.AdamW(second.planner_parameters(),lr=5e-6,weight_decay=0.)
    groups=[[(r,[dict(prompt=[1,2],tokens=[i,3],behavior=[-1.1,-2.2])]) for i,r in enumerate([1.14,0.06,0.02,1.12])]]
    assert grpo.update(first,opt1,groups)
    turns=grpo.weighted_turns(groups)
    with tempfile.TemporaryDirectory() as tmp,patch.object(parallel_replay,"helpers",return_value=None):
        p=Path(tmp)
        parallel_replay.compute(second,turns,p,2,"adapter",lock,write)
        # The trainer recovers the already written per-turn gradients.
        with patch.object(grpo,"backward_turn",side_effect=AssertionError("duplicate backward")):
            assert parallel_replay.apply(second,opt2,turns,p,2,"adapter",lock,write)
    for p,q in zip(first.planner_parameters(),second.planner_parameters(),strict=True):
        assert torch.equal(p,q)
        for k,x in opt1.state[p].items():
            assert torch.equal(x,opt2.state[q][k]),k
    print("Identical parameters and AdamW moments after resuming saved action gradients.")


if __name__=="__main__":
    main()
