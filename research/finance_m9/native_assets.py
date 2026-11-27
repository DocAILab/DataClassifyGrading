"""Assemble and verify the public files used by the native M9 controller."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXPERIMENTS = ROOT.parent / "finance_rl_v1" / "experiments"
B80 = "finance_program_student_b80_eval_20261021_r2"
RUNTIME = Path("vendor/finance_program_student_grpo_20261017_r3/runtime")


def read(path):
    return json.loads(Path(path).read_bytes())


def prepare(destination, scopes):
    """Use a private temporary directory per process. Preserve archived files."""
    destination = Path(destination)
    archived = EXPERIMENTS / "dependencies"
    for source in archived.rglob("*.py"):
        target = destination / source.relative_to(archived)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes().replace(b"\r\n", b"\n"))
    expected = read(ROOT / "assets/vendor-hashes.json")
    base = destination / B80
    public_runtime = ROOT.parent / "finance_program_v1/evaluation/runtime"
    for name, digest in expected.items():
        target = base / name
        if not target.exists():
            relative = Path(name).relative_to(RUNTIME)
            source = (ROOT / "assets/runtime-manifest.json" if relative.as_posix() ==
                      "reviews/runtime-manifest.json" else public_runtime / relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes().replace(b"\r\n", b"\n"))
        data = target.read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            # Two archived text assets used CRLF. Git checkouts may normalize them.
            original = data.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
            if hashlib.sha256(original).hexdigest() != digest:
                raise ValueError(f"Unexpected native runtime file: {name}")
            target.write_bytes(original)
    manifest = base / "runtime-inputs/manifest.json"
    manifest.parent.mkdir(exist_ok=True)
    manifest.write_text(json.dumps({"vendor_files_sha256": expected}), encoding="utf-8")
    scoped = destination / "finance_program_recovery_source_scope_20261024_r9/source-scopes.json"
    scoped.write_text(json.dumps({"rows": scopes}, ensure_ascii=False), encoding="utf-8")
    return destination
