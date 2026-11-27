"""Request fresh teacher replies under each retained field's recorded protocol."""
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parents[2] / "finance_teacher"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("indices", nargs="+", type=int)
    args = parser.parse_args()
    protocols = json.loads((HERE/"protocols.json").read_bytes())
    unknown = [i for i in args.indices if str(i) not in protocols["by_source_index"]]
    if unknown:
        parser.error(f"Source index outside the released 200 fields: {unknown}")
    with tempfile.TemporaryDirectory() as temp:
        worker = Path(temp)
        shutil.copytree(PACKAGE / "runtime", worker / "runtime", ignore=shutil.ignore_patterns("__pycache__"))
        (worker / "inputs").mkdir()
        shutil.copyfile(HERE / "cohort.json.gz", worker / "inputs/cohort.json.gz")
        shutil.copyfile(HERE / "runtime-sha256.json", worker / "runtime-sha256.json")
        collector = worker / "runtime/collector"
        for index in args.indices:
            output = args.output.resolve() / f"field-{index:03d}"
            output.mkdir(parents=True, exist_ok=False)
            name = protocols["by_source_index"][str(index)]
            plan = dict(protocols["plans"][name], source_indices=[index])
            (output/"plan.json").write_text(json.dumps(plan, ensure_ascii=False), encoding="utf-8")
            subprocess.run([sys.executable, "-B", str(collector/"smoke.py"), "run",
                            str(output/f"source-{index:03d}"), "--source-index", str(index)], check=True)
            subprocess.run([sys.executable, "-B", str(collector/"score_smoke.py"), str(output),
                            "--references", str(HERE/"references.json")], check=True)


if __name__ == "__main__":
    main()
