"""Evaluate the student SFT policy on the complete, label-blind Val422 cohort."""
from collections import Counter
from contextlib import ExitStack
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import eval_smoke as pilot

ROOT = pilot.ROOT
INPUT = pilot.M9 / "val422/val422-inputs.json"
ARMS = ("student",)
EXPECTED = {
    "student": "1d527f90b632a80809d6b0e152d62e04c773711c9ceb73ae8b441d7cb2f0bcd0",
    "skill": "0e52b9439b0016b0cad2903392030fca4b501019c7378a3a594b0fb6d1d4faf7",
    "input": "cd385146f8509023440fddcb6277e80a3f64dc6538e309fef0fa41486703db4f",
}
BATCH = 24


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.partial")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    os.replace(tmp, path)


def lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    file = path.open("a")
    try:
        fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        file.close()
        return None
    return file


def sources():
    if digest(INPUT) != EXPECTED["input"]:
        raise ValueError("VAL input checksum mismatch")
    data = read(INPUT)
    ids = [r["source_id"] for r in data["rows"]]
    if len(ids) != 422 or len(set(ids)) != 422 or data["split"] != "VAL":
        raise ValueError("Expected 422 distinct VAL fields")
    return ids


def receipt(arm, sid):
    """The final row is published only after checking the recorded native rollout."""
    folder = ROOT/"runs"/arm/sid
    result = read(folder/"result.json")
    trace = read(folder/"trace.json")
    calls = read(folder/"calls.json")
    if (sid != result["source_id"] or sid != trace["source_id"] or result["planner"] != EXPECTED[arm]
            or result["skill"] != EXPECTED["skill"] or calls[0]["stage"] != "Planner"
            or result["stop_reason"] != trace["stop_reason"] or result["prediction"] != trace["final_result"]
            or any("end" not in event for event in calls)
            or result["calls"] != sum(3 if e["stage"].startswith("SEARCH_STANDARD_") else 1 for e in calls)):
        raise ValueError(f"Invalid native result or call ledger for {arm}/{sid}")
    first = trace["planner_attempts"][0]
    try:
        first_tools = [s["tool"] for s in json.loads(first["raw_response"])["program"]]
    except (KeyError, TypeError, ValueError):
        first_tools = None
    return dict(**result, first_input_sha256=calls[0]["input_sha256"], first_tools=first_tools,
                path=[s["call"]["tool"] for p in trace["programs"] for s in p["steps"]],
                attempts=[a["status"] for a in trace["planner_attempts"]],
                trace_sha256=digest(folder/"trace.json"), calls_sha256=digest(folder/"calls.json"))


def work(arm, uuid):
    ids = sources()
    def claimed():
        processed = 0
        for sid in ids:
            summary = ROOT/"records"/arm/(sid+".json")
            review = ROOT/"review"/arm/(sid+".json")
            if summary.exists() or review.exists():
                continue
            claim = lock(ROOT/"locks"/arm/sid)
            if claim is None:
                continue
            with claim:
                if summary.exists() or review.exists():
                    continue
                folder = ROOT/"runs"/arm/sid
                # Isolate files from interrupted attempts before starting another rollout.
                if folder.exists() and not (folder/"result.json").is_file():
                    dest = ROOT/"prior-attempts"/arm/sid/str(time.time_ns())
                    dest.parent.mkdir(parents=True,exist_ok=True)
                    folder.rename(dest)
                write(ROOT/"workers"/(uuid+".json"),dict(arm=arm,source_id=sid,at=time.time(),pid=os.getpid()))
                if not (folder/"result.json").exists():
                    yield sid
                if (folder/"result.json").exists():
                    write(summary,receipt(arm,sid))
                else:
                    err = read(folder/"error.json")
                    write(review,dict(arm=arm,source_id=sid,gpu=uuid,at=time.time(),**err))
                    if "out of memory" in err["traceback"].lower():
                        return
                processed += 1
                if processed >= BATCH:
                    return
    pilot.evaluate(arm, source_ids=claimed(), output=ROOT/"runs")


def counts(directory, arm):
    return len(list((ROOT/directory/arm).glob("*.json")))


def queue():
    ids = sources()
    pending = {}
    last_inspection = 0
    cooldown = {}
    with ExitStack() as stack:
        owner = lock(ROOT/"queue.lock")
        if owner is None:
            return
        stack.enter_context(owner)
        while True:
            now = time.time()
            arms = ARMS
            for gpu,(process,arm,card,log) in list(pending.items()):
                rc = process.poll()
                if rc is None:
                    continue
                card.close()
                log.close()
                del pending[gpu]
                if rc and arm in arms:
                    write(ROOT/"worker-errors"/f"{time.time_ns()}.json",dict(gpu=gpu,arm=arm,exit_code=rc,at=now))
                    cooldown[gpu]=now+90
            done={a:counts("records",a) for a in arms}
            review={a:counts("review",a) for a in arms}
            running=Counter(arm for _,arm,_,_ in pending.values())
            state=dict(at=now,arms=arms,done=done,review=review,workers={g:a for g,(_,a,_,_) in pending.items()})
            write(ROOT/"status.json",state)
            if not last_inspection or now-last_inspection>=1800:
                write(ROOT/"inspections"/f"{int(now)}.json",state)
                last_inspection=now
            if all(done[a]==len(ids) for a in arms) and not pending:
                write(ROOT/"complete.json",state)
                return
            try:
                free=pilot.idle()
            except (OSError,subprocess.SubprocessError) as exc:
                write(ROOT/"gpu-query-error.json",dict(at=now,error=repr(exc)))
                free=[]
            for gpu in free:
                eligible=[a for a in arms if len(ids)-done[a]-review[a]>running[a]]
                if not eligible or gpu in pending or cooldown.get(gpu,0)>now:
                    continue
                card=lock(Path.home()/(".finance-student-sft-"+gpu+".lock"))
                if card is None:
                    continue
                if gpu not in pilot.idle():
                    card.close()
                    continue
                arm=min(eligible,key=lambda a:done[a]+BATCH*running[a])
                log=(ROOT/"logs"/f"{gpu}.log")
                log.parent.mkdir(exist_ok=True)
                log=log.open("ab")
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,PYTHONUNBUFFERED="1",PYTORCH_ALLOC_CONF="expandable_segments:True")
                process=subprocess.Popen([sys.executable,str(Path(__file__)),arm,gpu],
                    stdout=log,stderr=subprocess.STDOUT,env=env,pass_fds=(card.fileno(),))
                pending[gpu]=(process,arm,card,log)
                running[arm]+=1
            time.sleep(15)


if __name__=="__main__":
    if sys.argv[1:]==["queue"]:
        try:
            queue()
            if (ROOT/"complete.json").is_file() and not (ROOT/"评分.json").exists():
                with ExitStack() as stack:
                    owner = lock(ROOT/"finish.lock")
                    if owner is not None:
                        stack.enter_context(owner)
                        if not (ROOT/"评分.json").exists():
                            sys.path.insert(0,str(pilot.STUDY))
                            from archive_run import finish
                            finish("sft", ROOT)
        except Exception:
            write(ROOT/"queue-error.json",dict(at=time.time(),traceback=traceback.format_exc()))
            raise
    else:
        work(sys.argv[1],sys.argv[2])
