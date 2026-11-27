"""On-policy M9 campaign. Every sample and every optimizer boundary is restartable."""
from contextlib import contextmanager
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import patch

from skill_reward import skill_reward

CODE = Path(__file__).resolve().parent
STUDY = CODE.parent
M9 = STUDY.parent
sys.path.insert(0, str(M9))
ROOT = Path(os.environ.get("FINANCE_RL_RUN", STUDY / "work" / "rl")).resolve()
BASE = Path(os.environ.get("FINANCE_BASE_MODEL", "Qwen3.5-9B")).resolve()
PARENT = Path(os.environ.get("FINANCE_SFT_ADAPTER", STUDY / "weights" / "sft")).resolve()
SKILL = Path(os.environ.get("FINANCE_SKILL_ADAPTER", STUDY / "weights" / "skill")).resolve()
PLAN = json.loads((CODE / "plan.json").read_bytes())


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def write(path, data):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f".{os.getpid()}.partial")
    tmp.write_text(json.dumps(data,ensure_ascii=False),encoding="utf-8")
    os.replace(tmp,path)


@contextmanager
def lock(path):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("a") as f:
        try:
            fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            yield None
        else:
            yield f


def jobs(stage):
    if stage=="pilot":
        ids=PLAN["pilot"]
    elif stage.startswith("train-"):
        i=int(stage.split("-")[1])-1
        ids=PLAN["train"][8*i:8*i+8]
    elif stage.startswith("dev-"):
        ids=PLAN["dev"]
    elif stage=="val":
        ids=[r["source_id"] for r in read(M9/"val422/val422-inputs.json")["rows"]]
    else:
        raise ValueError(stage)
    return [(sid,k) for sid in ids for k in range(4 if stage=="pilot" or stage.startswith("train-") else 1)]


def folder(stage,sid,k):
    return ROOT/"runs"/stage/(sid+f"_{k}")


def results(stage):
    return [read(folder(stage,sid,k)/"result.json") for sid,k in jobs(stage)]


def checkpoint(n):
    return ROOT/"updates"/f"batch-{n:02d}"


def version_for(stage):
    if stage=="val":
        return read(ROOT/"selection.json")["batch"]
    if "-" not in stage:
        return 0
    return int(stage.split("-")[1])-(1 if stage.startswith("train-") else 0)


def identity(n):
    return PLAN["parent"] if not n else read(checkpoint(n)/"checkpoint.json")["planner_sha256"]


def get_backend(n, training=False):
    from model import load_backend, sha
    from transformers import set_seed
    set_seed(PLAN["seed"])
    assert sha(PARENT/"adapter_model.safetensors")==PLAN["parent"]
    assert sha(SKILL/"adapter_model.safetensors")==PLAN["skill"]
    args=SimpleNamespace(model=BASE,planner=PARENT,skill=SKILL,candidate=checkpoint(n)/"planner" if n else None)
    return load_backend(args,training=training)


def sample_worker(stage):
    import episode
    from native_assets import prepare
    from transformers import set_seed
    n=version_for(stage)
    val=stage=="val"
    public=episode.cohort(M9/"val422/val422-inputs.json" if val else CODE/"train-public.json","VAL" if val else "TRAIN")
    cases={c["source_id"]:c for c in public["inputs"]["cases"]}
    stochastic=stage=="pilot" or stage.startswith("train-")
    backend=None
    with tempfile.TemporaryDirectory() as tmp:
        episode.install_native(prepare(Path(tmp)/"deps",public["scopes"]))
        for sid,k in jobs(stage):
            dest=folder(stage,sid,k)
            if (dest/"result.json").exists():
                continue
            with lock(dest/"claim.lock") as owned:
                if owned is None or (dest/"result.json").exists():
                    continue
                if backend is None:
                    backend=get_backend(n)
                seed=int(digest(f"{PLAN['seed']}/{stage}/{sid}/{k}".encode())[:8],16)
                set_seed(seed)
                trace_box=[]
                original=episode.environment
                def capture(*args):
                    env=original(*args)
                    run=env.run
                    def capture_run(*inner):
                        trace,audit=run(*inner)
                        trace_box.append(trace)
                        return trace,audit
                    env.run=capture_run
                    return env
                with patch.object(episode,"environment",capture):
                    result,actions=episode.run_episode(public,cases[sid],backend,sample=stochastic,record_token_limit=True)
                trace=trace_box[0]
                if stochastic and not actions:
                    raise ValueError("No sampled Planner tokens")
                steps=[x for p in trace["programs"] for x in p["steps"]]
                tools=[x["call"]["tool"] for x in steps]
                bundle={"trace":trace,"actions":actions}
                payload=gzip.compress(json.dumps(bundle,ensure_ascii=False,separators=(",",":")).encode(),compresslevel=3,mtime=0)
                temp=dest/"evidence.partial"
                temp.write_bytes(payload)
                os.replace(temp,dest/"evidence.json.gz")
                write(dest/"result.json",dict(**result,source_id=sid,sample=k,seed=seed,batch=n,actor=identity(n),
                                              path=tools,sha256=digest(payload),attempts=len(trace["planner_attempts"])))


def rng_state():
    import torch
    import numpy as np
    return dict(python=random.getstate(),numpy=np.random.get_state(),cpu=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state())


def restore_rng(state):
    import torch
    import numpy as np
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["cpu"])
    torch.cuda.set_rng_state(state["cuda"])


def contract():
    return {name: digest((CODE / name).read_bytes())
            for name in ("plan.json", "train-public.json", "reward-labels.json", "skill_reward.py")}


def prepare_long_replay(backend):
    """Use the SFT memory bounds on unpadded replay, keeping policy dropout disabled."""
    import torch
    from teacher_memory import enable_bounded_delta
    model=backend.model
    enable_bounded_delta(model)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    model.enable_input_require_grads()
    model.train()
    for module in model.modules():
        if isinstance(module,torch.nn.Dropout):
            module.eval()
    # policy_math passes an all-ones mask for exactly one unpadded row.
    def unpadded(module,args,kwargs):
        mask=kwargs.get("attention_mask")
        if mask is not None and mask.shape[0]==1 and torch.all(mask==1).item():
            kwargs["attention_mask"]=None
        return args,kwargs
    model.register_forward_pre_hook(unpadded,with_kwargs=True)


def training_groups(batch):
    stage=f"train-{batch:02d}"
    labels=read(CODE/"reward-labels.json")
    metadata={r["source_id"]:r["metadata"] for r in read(CODE/"train-public.json")["rows"]}
    groups=[]
    breakdown=[]
    for sid,k in jobs(stage):
        dest=folder(stage,sid,k)
        row=read(dest/"result.json")
        assert (row["source_id"],row["sample"],row["actor"],row["batch"])==(sid,k,identity(batch-1),batch-1)
        payload=(dest/"evidence.json.gz").read_bytes()
        if digest(payload)!=row["sha256"]:
            raise ValueError("Damaged rollout")
        if not k:
            groups.append([])
        decoded=json.loads(gzip.decompress(payload))
        parts=skill_reward(row,decoded["trace"],labels[sid],metadata[sid])
        assert parts["outcome"]==int(row["valid"] and row["prediction"]==labels[sid])
        reward=parts["total"]
        breakdown.append(dict(source_id=sid,sample=k,**parts))
        groups[-1].append((reward,decoded["actions"]))
    return groups,breakdown


def backward_worker(batch):
    from grpo import weighted_turns
    from parallel_replay import compute
    backend=get_backend(batch-1,training=True)
    prepare_long_replay(backend)
    groups,_=training_groups(batch)
    compute(backend,weighted_turns(groups),ROOT,batch,identity(batch-1),lock,write)


def update_worker(batch):
    import torch
    from model import sha
    from grpo import update
    if checkpoint(batch).exists():
        return
    with lock(ROOT/"update.lock") as owned:
        if owned is None or checkpoint(batch).exists():
            return
        backend=get_backend(batch-1,training=True)
        prepare_long_replay(backend)
        optimizer=torch.optim.AdamW(backend.planner_parameters(),lr=5e-6,weight_decay=0.)
        if batch>1:
            prev=checkpoint(batch-1)
            receipt=read(prev/"checkpoint.json")
            if receipt["next_batch"]!=batch or receipt["contract"]!=contract() or sha(prev/"resume.pt")!=receipt["resume_sha256"]:
                raise ValueError("Checkpoint cursor or training contract differs")
            state=torch.load(prev/"resume.pt",map_location="cpu",weights_only=False)
            optimizer.load_state_dict(state["optimizer"])
            restore_rng(state["rng"])
            write(ROOT/"restores"/f"batch-{batch:02d}.json",dict(previous=batch-1,resume_sha256=receipt["resume_sha256"],
                  optimizer_states=len(optimizer.state),at=time.time(),pid=os.getpid()))
        groups,breakdown=training_groups(batch)
        def record_turn(record):
            write(ROOT/"replay-progress"/f"batch-{batch:02d}.json",dict(at=time.time(),**record))
        if batch >= 10:
            from grpo import weighted_turns
            from parallel_replay import apply
            changed=apply(backend,optimizer,weighted_turns(groups),ROOT,batch,identity(batch-1),lock,write)
        else:
            changed=update(backend,optimizer,groups,on_turn=record_turn)
        out=checkpoint(batch)
        out.parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=out.parent) as temp:
            p=Path(temp)/"boundary"
            backend.model.save_pretrained(p,selected_adapters=["planner"])
            torch.save(dict(optimizer=optimizer.state_dict(),rng=rng_state()),p/"resume.pt")
            weight=sha(p/"planner/adapter_model.safetensors")
            # PEFT may reserialize equivalent tensors. Verify actual tensor changes.
            from safetensors.torch import load_file
            before=load_file(str((PARENT if batch==1 else checkpoint(batch-1)/"planner")/"adapter_model.safetensors"))
            after=load_file(str(p/"planner/adapter_model.safetensors"))
            assert set(before)==set(after) and all(torch.isfinite(x).all() for x in after.values())
            norm=sum((after[k].double()-v.double()).square().sum().item() for k,v in before.items())**.5
            if (norm>0)!=changed:
                raise RuntimeError("Optimizer update disagrees with measured weight change")
            write(p/"checkpoint.json",dict(batch=batch,next_batch=batch+1,parent=PLAN["parent"],fixed_skill=PLAN["skill"],
                  planner_sha256=weight,resume_sha256=sha(p/"resume.pt"),contract=contract(),updated=changed,
                  delta_norm=norm,rewards=[[r for r,a in group] for group in groups],optimizer_states=len(optimizer.state),
                  **({"skill_reward_components":breakdown})))
            p.rename(out)
        if changed and not (ROOT/"restart-test.json").exists():
            write(ROOT/"restart-test.json",dict(after_batch=batch,checkpoint=weight,pid=os.getpid(),at=time.time()))
            os._exit(75)


def score(rows):
    labels=read(CODE/"reward-labels.json")
    return sum(r["valid"] and r["prediction"]==labels[r["source_id"]] for r in rows)


def next_stage():
    for stage in ["pilot","dev-00"]+[item for b in range(1,17) for item in
                 ([f"train-{b:02d}",f"update-{b:02d}"]+([f"dev-{b:02d}"] if b in PLAN["dev_batches"] else []))]+["val"]:
        if stage.startswith("update"):
            if not checkpoint(int(stage.split("-")[1])).exists():
                return stage
            continue
        if stage=="val" and not (ROOT/"selection.json").exists():
            scores={b:score(results(f"dev-{b:02d}")) for b in [0,*PLAN["dev_batches"]]}
            trained=[b for b in PLAN["dev_batches"] if any(read(checkpoint(i)/"checkpoint.json")["updated"] for i in range(1,b+1))]
            if not trained:
                raise ValueError("No real update; review pilot and training groups")
            winner=max(trained,key=lambda b:(scores[b],-b))
            write(ROOT/"selection.json",dict(batch=winner,scores=scores,at=time.time(),rule="highest trained development joint score, earliest tie; retain original baseline for comparison"))
        if not all((folder(stage,s,k)/"result.json").exists() for s,k in jobs(stage)):
            return stage
        if stage=="pilot" and not (ROOT/"pilot-summary.json").exists():
            rows=results(stage)
            scores=[score(rows[i:i+4]) for i in range(0,len(rows),4)]
            mixed=sum(0<x<4 for x in scores)
            write(ROOT/"pilot-summary.json",dict(fields=len(scores),mixed=mixed,group_scores=scores,correct=sum(scores),
                  different_routes=sum(len({tuple(r["path"]) for r in rows[i:i+4]})>1 for i in range(0,len(rows),4))))
        if stage=="pilot" and not read(ROOT/"pilot-summary.json")["mixed"] and not (ROOT/"pilot-review.json").exists():
            return "review-pilot"
    return "complete"


def queue():
    from gpu_resources import idle
    os.environ.setdefault("FINANCE_ALLOWED_GPUS", os.environ.get("CUDA_VISIBLE_DEVICES", "all"))
    with lock(ROOT/"queue.lock") as owner:
        if owner is None:
            return
        pending={}
        last_stage=None
        while True:
            for gpu,(proc,log,card,worker_stage) in list(pending.items()):
                code=proc.poll()
                if code is None:
                    continue
                log.close()
                card.close()
                del pending[gpu]
                tested=read(ROOT/"restart-test.json") if (ROOT/"restart-test.json").exists() else {}
                expected_restart=code==75 and tested.get("pid")==proc.pid and worker_stage==f"update-{tested.get('after_batch',0):02d}"
                if code and not expected_restart:
                    write(ROOT/"errors"/f"{time.time_ns()}.json",dict(stage=worker_stage,gpu=gpu,exit=code,log=log.name))
            if list((ROOT/"errors").glob("*.json")):
                stage="review-error"
            else:
                stage=next_stage()
            if stage!=last_stage:
                print(time.strftime("%F %T"),stage,flush=True)
                last_stage=stage
            active={g:s for g,(_,_,_,s) in pending.items()}
            write(ROOT/"status.json",dict(stage=stage,workers=active,at=time.time()))
            if stage=="complete" and not pending:
                write(ROOT/"complete.json",dict(at=time.time(),selected=read(ROOT/"selection.json")))
                return
            if not stage.startswith("review") and stage!="complete":
                for gpu in idle():
                    if gpu in pending or (stage.startswith("update") and pending):
                        continue
                    card=(Path.home()/(".finance-student-sft-"+gpu+".lock")).open("a")
                    try:
                        fcntl.flock(card,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError:
                        card.close()
                        continue
                    if gpu not in idle():
                        card.close()
                        continue
                    logdir=ROOT/"logs"
                    logdir.mkdir(exist_ok=True)
                    log=(logdir/f"{stage}-{gpu[-6:]}-{time.time_ns()}.log").open("w")
                    proc=subprocess.Popen([sys.executable,"-u",str(Path(__file__)),stage],cwd=ROOT,stdout=log,stderr=log,
                        env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS="2",TOKENIZERS_PARALLELISM="false"),
                        pass_fds=(card.fileno(),))
                    pending[gpu]=(proc,log,card,stage)
            time.sleep(12)


if __name__=="__main__":
    stage=sys.argv[1]
    if stage=="queue":
        queue()
        if (ROOT/"complete.json").is_file() and not (ROOT/"评分.json").exists():
            with lock(ROOT/"finish.lock") as owner:
                if owner is not None and not (ROOT/"评分.json").exists():
                    sys.path.insert(0, str(STUDY))
                    from archive_run import finish
                    finish("rl", ROOT)
    elif stage.startswith("update-"):
        update_worker(int(stage.split("-")[1]))
    elif stage.startswith("backward-"):
        backward_worker(int(stage.split("-")[1]))
    else:
        sample_worker(stage)
