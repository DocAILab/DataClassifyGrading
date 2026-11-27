"""Single-job queue on aiss-3. Wait for an atomically published final dataset and an idle 6000."""
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent
HOME = Path(os.environ.get("FINANCE_TRAIN_DIR", ROOT.parent/"work/sft")).resolve()
M9 = ROOT.parents[1]
BASE = Path(os.environ.get("FINANCE_BASE_MODEL", "Qwen3.5-9B")).resolve()
DATA = ROOT / "planner.jsonl"
OUTPUT = HOME / "planner-adapter"
INSPECTIONS = HOME / "inspections"
INTERVAL = 1800


def status(phase, **values):
    dest = HOME / "gpu-status.json"
    INSPECTIONS.mkdir(exist_ok=True)
    now = time.time()
    data = dict(phase=phase, at=now, watchdog_pid=os.getpid(), **values)
    last = INSPECTIONS / "last.json"
    previous = json.loads(last.read_bytes()) if last.exists() else {}
    if (phase != previous.get("phase") or now - previous.get("at", 0) >= INTERVAL):
        report = INSPECTIONS / f"{time.time_ns()}-{phase}.json"
        report.write_text(json.dumps(data))
        tmp = last.with_suffix(".partial")
        tmp.write_text(json.dumps(data))
        tmp.replace(last)
        previous = data
    data["last_inspection"] = previous.get("at")
    tmp = dest.with_suffix(".partial")
    tmp.write_text(json.dumps(data))
    tmp.replace(dest)


def lock(path):
    stream = path.open("a")
    try:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        stream.close()
        return None
    return stream


def idle_cards():
    sys.path.insert(0, str(ROOT.parent/"rl"))
    from gpu_resources import idle
    return idle()


def read_log(offset=0):
    log = HOME / "training.log"
    if not log.exists():
        return ""
    with log.open("rb") as stream:
        stream.seek(max(offset, log.stat().st_size - 32768))
        return stream.read().decode(errors="replace")


def record_failure(info, tail):
    """Retry interruptions and diagnosed OOMs, preserving failure timestamps."""
    if "torch.OutOfMemoryError: CUDA out of memory" not in tail:
        if info["exit_code"] in (-1, -9, -15, 129, 137, 143) and not (HOME/"pause-requested").exists():
            with (HOME/"interruptions.jsonl").open("a") as out:
                out.write(json.dumps(dict(**info, progress=activity())) + "\n")
            status("retry_interrupted", reason="训练进程中断，等待从已保存的更新继续", **info)
            return True
        status("training_failed", reason=tail[-650:], **info)
        return False
    other = re.search(r"Process \d+ has ([\d.]+ [GM]iB) memory in use", tail)
    errors = HOME / "oom-history.jsonl"
    history = [json.loads(x) for x in errors.read_bytes().splitlines()] if errors.exists() else []
    shared = bool(other)
    # A second process can arrive even after several empty-card checks. Use the
    # verified smaller-memory kernel on retry regardless of who owned the RAM.
    if not shared and info.get("bounded_delta"):
        status("training_failed", reason="独占显卡上分段计算仍显存不足，需检查训练日志", **info)
        return False
    if not any(r["failed_at"] == info["failed_at"] for r in history):
        with errors.open("a") as out:
            out.write(json.dumps(dict(**info, other_gpu_memory=other.group(1) if other else None,
                                      use_bounded_delta=True)) + "\n")
    status("retry_oom", reason="显存冲突，重新等待空卡" if shared else "显存不足，下次使用分段计算",
           **info)
    return True


def activity():
    path = HOME / "train-progress.json"
    return json.loads(path.read_bytes()) if path.exists() else {"updates": 0}


def audit(path, preview):
    from transformers import AutoTokenizer
    sys.path.insert(0, str(M9))
    from train_sft import teacher_examples
    tok = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
    limit = json.loads((BASE / "config.json").read_bytes())["text_config"]["max_position_embeddings"]
    examples = teacher_examples(path, tok, limit, preview=preview)
    return dict(rows=len(examples), longest=max(len(x) for x, _ in examples),
                shortest=min(len(x) for x, _ in examples), context_limit=limit)


def verify_output():
    record = json.loads((OUTPUT / "sft.json").read_bytes())
    final = json.loads(DATA.with_name("manifest.json").read_bytes())
    weights = OUTPUT / "adapter_model.safetensors"
    if (record.get("teacher") != final or final["sources"] != 200 or final["preview"]
            or record["component"] != "planner" or record["data_sha256"] != final["data_sha256"]
            or record["updates"] != (final["rows"] + 15) // 16
            or hashlib.sha256(weights.read_bytes()).hexdigest() != record["sha256"]):
        raise ValueError("Saved adapter failed verification")
    status("completed", updates=record["updates"], adapter=str(OUTPUT), sha256=record["sha256"])


def main():
    HOME.mkdir(parents=True,exist_ok=True)
    owner = lock(HOME / "queue.lock")
    if not owner:
        return
    if OUTPUT.exists():
        verify_output()
        return
    failure = HOME / "training-failure.json"
    if failure.is_file():
        info = json.loads(failure.read_bytes())
        info.setdefault("failed_at", failure.stat().st_mtime)
        if not record_failure(info, read_log(info.get("offset", 0))):
            return
        failure.rename(HOME / f"requeued-{time.time_ns()}.json")
    while not DATA.with_name("manifest.json").exists():
        status("waiting_teacher")
        time.sleep(60)
    status("checking_final_dataset")
    token_audit = audit(DATA, preview=False)
    (HOME / "final-token-audit.json").write_text(json.dumps(token_audit))
    idle_since = {}
    history = HOME / "oom-history.jsonl"
    while True:
        retries = [json.loads(x) for x in history.read_bytes().splitlines()] if history.exists() else []
        audit_state = dict(**token_audit, retries=len(retries), progress=activity())
        try:
            idle = idle_cards()
            now = time.monotonic()
            idle_since = {uuid: idle_since.get(uuid, now) for uuid in idle}
            status("waiting_idle_gpu", idle_cards=len(idle), **audit_state)
            for uuid in idle:
                if now - idle_since[uuid] < 120:
                    continue
                card = lock(Path.home() / (".finance-student-sft-" + uuid + ".lock"))
                if card is None:
                    continue
                with card:
                    if uuid not in idle_cards():
                        continue
                    bounded = bool(retries)
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, PYTHONUNBUFFERED="1",
                               FINANCE_BOUNDED_DELTA=str(int(bounded)), FINANCE_TRAIN_DIR=str(HOME),
                               PYTORCH_ALLOC_CONF="expandable_segments:True")
                    args = [sys.executable, str(M9 / "train_sft.py"), "teacher-planner", "--model", str(BASE),
                            "--data", str(DATA), "--output", str(OUTPUT)]
                    with (HOME / "training.log").open("ab") as log:
                        offset = log.tell()
                        # Inherit the locks so an interrupted supervisor cannot start a duplicate training process.
                        child = subprocess.Popen(args, env=env, stdout=log, stderr=subprocess.STDOUT,
                                                 pass_fds=(owner.fileno(), card.fileno()))
                        while True:
                            audit_state["progress"] = activity()
                            status("training", gpu=uuid, pid=child.pid, bounded_delta=bounded, **audit_state)
                            try:
                                rc = child.wait(timeout=60)
                                break
                            except subprocess.TimeoutExpired:
                                pass
                    if rc:
                        info = dict(exit_code=rc, gpu=uuid, log=str(HOME / "training.log"),
                                    offset=offset, failed_at=time.time(), bounded_delta=bounded)
                        failure.write_text(json.dumps(info))
                        if not record_failure(info, read_log(offset)):
                            return
                        failure.rename(HOME / f"requeued-{time.time_ns()}.json")
                        idle_since.clear()
                        break
                    verify_output()
                    return
        except (subprocess.SubprocessError, FileNotFoundError) as error:
            status("gpu_query_retry", reason=str(error))
        time.sleep(60)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--preview":
        print(json.dumps(audit(Path(sys.argv[2]), preview=True)))
    else:
        try:
            main()
        except Exception as error:
            status("queue_error", reason=str(error))
            raise
