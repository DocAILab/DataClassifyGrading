"""Seal a completed GPU run, then score it offline using the frozen Val422 references."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile

STUDY = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_bytes())


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def finish(kind, folder):
    run = Path(folder).resolve()
    complete = read(run / "complete.json")
    sys.path.insert(0, str(STUDY / kind))
    if kind == "sft":
        import score_full
        rows = [read(f) for f in sorted((run / "records/student").glob("*.json"))]
        if len(rows) != 422 or complete["arms"] != ["student"]:
            raise ValueError("Expected 422 completed student predictions")
        write(run / "snapshot.json", dict(finished=True, state=complete, records={"student": rows},
              review={"student": [read(f) for f in (run/"review/student").glob("*.json")]}))
        name, members = "全部轨迹.tar.gz", ["runs/student", "records/student", "complete.json"]
    else:
        import score
        fields = list((run/"runs/val").glob("*/result.json"))
        if len(fields) != 422 or list((run/"errors").glob("*.json")):
            raise ValueError("Expected 422 completed RL predictions and no pending errors")
        updates = [read(f) for f in sorted((run/"updates").glob("*/checkpoint.json"))]
        assert len(updates) == 16
        shutil.copyfile(STUDY/"rl/plan.json", run/"plan.json")
        write(run/"progress.json", dict(complete=complete, errors=[], pilot=read(run/"pilot-summary.json"),
              selected=read(run/"selection.json"), updates=updates, restart=read(run/"restart-test.json"),
              restores=[read(f) for f in sorted((run/"restores").glob("*.json"))]))
        name = "evidence.tar.gz"
        members = ["runs", "restores", "plan.json", "selection.json", "complete.json",
                   "pilot-summary.json", "restart-test.json", *(f"updates/batch-{i:02d}/checkpoint.json" for i in range(1,17))]
    with tempfile.TemporaryDirectory(dir=run) as temp:
        archive = Path(temp) / name
        with tarfile.open(archive, "w:gz", compresslevel=3) as tar:
            for member in members:
                tar.add(run/member, arcname=member)
        os.replace(archive, run/name)
    if kind == "sft":
        score_full.HERE = run
        score_full.main()
    else:
        score.main(run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["sft", "rl"])
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    finish(args.stage, args.run)
