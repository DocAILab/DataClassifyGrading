"""Distribute independent backward passes; one trainer accumulates them in sample order."""
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from threading import Event, Thread
import time


def shard_root(root,batch):
    return root/"gradients"/f"batch-{batch:02d}"


def compute(backend,turns,root,batch,actor,lock,write):
    import torch
    from grpo import backward_turn
    from model import sha as file_hash
    if not turns:
        return
    dest=shard_root(root,batch)
    denominator=sum(len(a["tokens"]) for a,_ in turns)
    names=[(n,p) for n,p in backend.model.named_parameters() if ".lora_" in n and ".planner." in n]
    for i,(action,adv) in enumerate(turns):
        key=f"turn-{i:04d}"
        receipt=dest/(key+".json")
        if receipt.exists():
            continue
        with lock(dest/(key+".lock")) as owner:
            if owner is None or receipt.exists():
                continue
            backend.model.zero_grad(set_to_none=True)
            loss=backward_turn(backend,action,adv,denominator)
            if not math.isfinite(loss):
                raise FloatingPointError("Nonfinite replay loss")
            gradients={n:p.grad.detach().cpu() for n,p in names if p.grad is not None}
            if len(gradients)!=len(names):
                raise ValueError("Missing Planner gradient")
            tmp=dest/(key+f".{os.getpid()}.partial")
            torch.save(gradients,tmp)
            sha=file_hash(tmp)
            os.replace(tmp,dest/(key+".pt"))
            write(receipt,dict(actor=actor,batch=batch,turn=i,denominator=denominator,advantage=adv,
                  sha256=sha,action_tokens=len(action["tokens"]),prompt_tokens=len(action["prompt"]),
                  loss=loss,at=time.time(),gpu=os.getenv("CUDA_VISIBLE_DEVICES"),pid=os.getpid()))


def helpers(root,batch,stop,lock):
    from gpu_resources import idle
    children={}
    attempted=set()
    while not stop.is_set():
        for gpu,(p,file,guard) in list(children.items()):
            if p.poll() is not None:
                guard.__exit__(None,None,None)
                file.close()
                del children[gpu]
        for gpu in idle():
            if stop.is_set() or gpu in attempted or gpu==os.environ.get("CUDA_VISIBLE_DEVICES"):
                continue
            guard=lock(Path.home()/(".finance-student-sft-"+gpu+".lock"))
            card=guard.__enter__()
            if card is None or gpu not in idle():
                guard.__exit__(None,None,None)
                continue
            file=(root/"logs"/f"backward-{batch:02d}-{gpu[-6:]}-{time.time_ns()}.log").open("w")
            try:
                p=subprocess.Popen([sys.executable,"-u",str(Path(__file__).with_name("campaign.py")),f"backward-{batch:02d}"],cwd=root,
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu),stdout=file,stderr=file,pass_fds=(card.fileno(),))
            except Exception:
                guard.__exit__(None,None,None)
                file.close()
                raise
            attempted.add(gpu)
            children[gpu]=(p,file,guard)
        stop.wait(10)
    for p,file,guard in children.values():
        p.wait()
        guard.__exit__(None,None,None)
        file.close()


def apply(backend,optimizer,turns,root,batch,actor,lock,write):
    import torch
    from model import sha as file_hash
    if not turns:
        return False
    dest=shard_root(root,batch)
    write(root/"replay-progress"/f"batch-{batch:02d}.json",dict(turn=len(list(dest.glob("turn-*.json"))),total=len(turns),at=time.time()))
    stop=Event()
    pool=Thread(target=helpers,args=(root,batch,stop,lock),daemon=True)
    pool.start()
    try:
        while True:
            compute(backend,turns,root,batch,actor,lock,write)
            done=len(list(dest.glob("turn-*.json")))
            write(root/"replay-progress"/f"batch-{batch:02d}.json",dict(turn=done,total=len(turns),at=time.time()))
            if done==len(turns):
                break
            time.sleep(5)
    finally:
        stop.set()
        pool.join()
    optimizer.zero_grad(set_to_none=True)
    named={n:p for n,p in backend.model.named_parameters() if ".lora_" in n and ".planner." in n}
    denominator=sum(len(a["tokens"]) for a,_ in turns)
    for i,(_,adv) in enumerate(turns):
        key=f"turn-{i:04d}"
        record=json.loads((dest/(key+".json")).read_bytes())
        weights=dest/(key+".pt")
        sha=file_hash(weights)
        if (record["actor"],record["batch"],record["turn"],record["denominator"],record["advantage"],record["sha256"])!=(actor,batch,i,denominator,adv,sha):
            raise ValueError("Gradient belongs to another action or policy")
        gradients=torch.load(weights,weights_only=True,map_location="cpu")
        if set(gradients)!=set(named):
            raise ValueError("Planner parameter names changed")
        for name,grad in gradients.items():
            p=named[name]
            value=grad.to(device=p.device,dtype=p.dtype)
            if p.grad is None:
                p.grad=value
            else:
                p.grad.add_(value)
    torch.nn.utils.clip_grad_norm_(list(named.values()),1.,error_if_nonfinite=True)
    optimizer.step()
    return True
