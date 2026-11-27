"""Pin the training, smoke and table-held-out development fields before sampling."""
import argparse
from collections import defaultdict
from zipfile import ZipFile
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
M9 = HERE.parents[1]
sys.path.insert(0, str(M9))
from resident_input_contract import clean_metadata


def read(path):
    return json.loads(path.read_bytes())


def order(value):
    return hashlib.sha256(("planner-rl-20261126/"+value).encode()).hexdigest()


def balanced(rows, count):
    groups = defaultdict(list)
    for row in sorted(rows, key=lambda r: order(r["id"])):
        groups[row["classification"]["category_leaf_level"]].append(row["id"])
    chosen = []
    while len(chosen) < count and any(groups.values()):
        for name in sorted(groups, key=order):
            if groups[name] and len(chosen)<count:
                chosen.append(groups[name].pop())
    return chosen


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("source", type=Path, help="Original finance split containing data/split_boundaries.json and raw/DCG_FIN_TRAIN.json")
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    DATA = args.source
    args.output.mkdir(parents=True, exist_ok=True)
    def write(name, data):
        # Preserve the bytes used to identify the original experiment on both Windows and Linux.
        text = json.dumps(data, ensure_ascii=False, indent=2)
        (args.output / name).write_bytes(text.replace("\n", "\r\n").encode())
    bounds = read(DATA/"data/split_boundaries.json")
    raw = (DATA/"raw/DCG_FIN_TRAIN.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest()==bounds["split_sha256"]["train"]
    rows = json.loads(raw)
    with ZipFile(HERE.parent/"teacher/traces.zip") as archive:
        teacher = {json.loads(line)["source_id"] for line in archive.read("index.jsonl").splitlines()}
    skill = set(read(HERE/"frozen-skill-train-ids.json"))
    ids = {r["id"] for r in rows}
    assert ids==set(bounds["ids"]["train"]) and not ids.intersection(bounds["ids"]["val"]+bounds["ids"]["test"])
    clean = [r for r in rows if r["id"] not in teacher|skill]
    assert len(clean)==402 and len(teacher)==200 and len(skill)==1169
    dev_tables = {"T_H_QSK","XT_BF_FIRMCARD","THQLOG","TSYSINFO_TODOLIST","TSYS_USER_MESSAGE"}
    dev_pool = [r for r in clean if r["source_table_name"] in dev_tables]
    train_pool = [r for r in clean if r["source_table_name"] not in dev_tables]
    dev = balanced(dev_pool, 41)
    # Each field in the pilot belongs to the training partition. Train uses current-policy draws.
    train = balanced(train_pool,128)
    pilot = train[:32]
    levels = {"常规一般数据":"L1","敏感一般数据":"L2","重要数据":"L3"}
    public = []
    refs = {}
    for row in rows:
        sid = row["id"]
        meta = clean_metadata(row["metadata"])
        meta.pop("value", None)
        public.append({"source_id":sid,"metadata":meta})
        if sid in set(train+dev):
            cat = row["classification"]
            refs[sid] = {"category":"finance:"+"/".join(cat["category_"+k+"_level"] for k in ("root","branch","leaf")),
                         "grading_level":levels[row["grading"]["sensitivity_level"]]}
    assert len(refs)==169 and not set(dev)&set(train)
    write("train-public.json", {"split":"TRAIN","rows":public})
    write("reward-labels.json",refs)
    write("plan.json",dict(pilot=pilot,train=train,dev=dev,dev_tables=sorted(dev_tables),train_pool=len(train_pool),
                           dev_pool=len(dev_pool),batch_sources=8,group_size=4,dev_batches=[4,8,16],seed=20261126,
                           parent="1d527f90b632a80809d6b0e152d62e04c773711c9ceb73ae8b441d7cb2f0bcd0",
                           skill="0e52b9439b0016b0cad2903392030fca4b501019c7378a3a594b0fb6d1d4faf7"))
    print("eligible",len(clean),"pilot",len(pilot),"train",len(train),"dev",len(dev))


if __name__=="__main__":
    main()
