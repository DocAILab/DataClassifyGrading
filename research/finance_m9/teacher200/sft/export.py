"""Rebuild the exact student supervision from recorded Planner requests and native traces."""
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

HERE = Path(__file__).resolve().parent
TEACHER = HERE.parent / "teacher"


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()


def export():
    manifest = json.loads((HERE / "manifest.json").read_bytes())
    archive = TEACHER / "traces.zip"
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == manifest["archive_sha256"]
    rows = []
    skipped = Counter()
    seen = set()
    with ZipFile(archive) as traces, gzip.open(TEACHER / "planner-wire.jsonl.gz", "rb") as requests:
        index = [json.loads(line) for line in traces.read("index.jsonl").splitlines()]
        for entry, line in zip(index, requests, strict=True):
            wire = json.loads(line)
            sid = entry["source_id"]
            assert sid == wire["source_id"] and sid not in seen
            seen.add(sid)
            raw = traces.read(entry["trace"])
            assert hashlib.sha256(raw).hexdigest() == entry["trace_sha256"]
            trace = json.loads(raw)
            assert trace["terminated"] and trace["source_id"] == sid
            programs = {p["attempt"]: p for p in trace["programs"]}
            ended = False
            for turn, (attempt, call) in enumerate(zip(trace["planner_attempts"], wire["calls"], strict=True), 1):
                assert attempt["attempt"] == turn
                program = programs.get(turn)
                if not program or attempt["status"] != "accepted" or program["status"] not in ("completed", "terminated"):
                    skipped["rejected_or_failed"] += 1
                    continue
                messages, reply = call["messages"], call["reply"]
                assert call["same_information_exact"] and call["role"] == "Planner" and call["complete"]
                assert hashlib.sha256(compact(messages)).hexdigest() == call["input_sha256"] == call["student_view_sha256"]
                assert hashlib.sha256(reply.encode()).hexdigest() == call["reply_sha256"]
                assert reply == attempt["raw_response"] and json.loads(reply) == program["plan"]
                assert not program["unexecuted_tail"] and all(step["status"] == "success" for step in program["steps"])
                assert [s["call"]["tool"] for s in program["steps"]] == [s["tool"] for s in program["plan"]["program"]]
                ended |= program["terminated"]
                rows.append(compact(dict(source_id=sid, turn=turn, messages=messages, completion=reply)))
            assert ended
    data = b"\n".join(rows) + b"\n"
    assert (len(seen), len(rows), dict(skipped)) == (manifest["sources"], manifest["rows"], manifest["skipped"])
    assert hashlib.sha256(data).hexdigest() == manifest["data_sha256"]
    out = HERE / "planner.jsonl"
    tmp = out.with_suffix(".partial")
    tmp.write_bytes(data)
    tmp.replace(out)
    print(f"{len(seen)} sources, {len(rows)} demonstrations, SHA-256 {manifest['data_sha256']}")


if __name__ == "__main__":
    export()
