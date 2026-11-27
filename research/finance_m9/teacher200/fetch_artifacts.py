"""Install the published evidence and checkpoints with SHA-256 verification."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from urllib.request import urlopen
from zipfile import ZipFile

HERE = Path(__file__).resolve().parent
TAG = "finance-teacher200-sft-grpo-20261127"
URL = "https://github.com/DocAILab/DataClassifyGrading/releases/download/" + TAG


def install(name, source):
    manifest = json.loads((HERE/"release-assets.json").read_bytes())[name]
    with tempfile.TemporaryDirectory(dir=HERE) as work:
        file = Path(work) / name
        if source:
            file = source / name
        else:
            with urlopen(f"{URL}/{name}", timeout=120) as remote, file.open("wb") as local:
                shutil.copyfileobj(remote, local)
        with file.open("rb") as stream:
            sha = hashlib.sha256()
            for block in iter(lambda: stream.read(1024*1024), b""):
                sha.update(block)
            digest = sha.hexdigest()
        if digest != manifest["sha256"] or file.stat().st_size != manifest["bytes"]:
            raise ValueError(f"Unexpected release asset: {name}")
        if name == "rl-evidence.tar.gz":
            target = HERE/"results/rl/evidence.tar.gz"
            partial = Path(work)/"verified.tar.gz"
            shutil.copyfile(file, partial)
            partial.replace(target)
        else:
            target = HERE/("work/sft" if name=="sft-checkpoints.zip" else "results/rl")
            with ZipFile(file) as archive:
                for info in archive.infolist():
                    destination = (target / info.filename).resolve()
                    if not destination.is_relative_to(target.resolve()) or info.is_dir():
                        raise ValueError(f"Unexpected checkpoint archive entry: {info.filename}")
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    partial = Path(work)/"verified-checkpoint"
                    with archive.open(info) as src, partial.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
                    partial.replace(destination)
        print(f"{name}  {digest}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-dir", type=Path, help="Use already downloaded release assets")
    parser.add_argument("--sft-checkpoints", action="store_true", help="Also install the 13 SFT optimizer boundaries")
    args=parser.parse_args()
    for filename in ("rl-evidence.tar.gz","rl-checkpoints.zip"):
        install(filename, args.from_dir)
    if args.sft_checkpoints:
        install("sft-checkpoints.zip", args.from_dir)
