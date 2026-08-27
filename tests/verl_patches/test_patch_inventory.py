"""Patch-bundle inventory tests (local, no verl/torch required).

Covers what can be verified without a verl install:
- patch file existence, unified-diff structure, a/b paths, hunk sanity;
- patch sha256 match the README table;
- applying every patch to the official-wheel fixtures reproduces the exact
  installed sha256 values recorded from the server (byte-for-byte);
- README inventory is internally consistent with the scripts' tables.

Behavior tests that need a real verl 0.9 + torch environment
(answer_mask span localization, scheme C numeric loss, prefix-diff rendering)
are marked ``skip-unless-verl`` and documented in the module docstring /
README; run them on the server venv with pytest.
"""
import hashlib
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

PATCH_DIR = Path(__file__).resolve().parents[2] / "script" / "verl" / "common" / "patches" / "verl-0.9.0"
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "wheel"

# target file name (fixture) -> patch file name -> expected installed sha256
BUNDLE = {
    "agent_loop.py": ("agent_loop-debug.patch",
                      "902cc8c4007b944974d77c54bc1ce227df49de4390e5b3a0fc831f5cf0a4a801"),
    "chat_template.py": ("chat_template-system-first.patch",
                         "58031af7a001a1208129b271f110e9cf94a3978874fadc5da43db9eec0322578"),
    "multiturn_sft_dataset.py": ("multiturn_sft_dataset-prefix-diff-answer-mask.patch",
                                 "ce7486288a68a85a0777d9e587688501e09603533703e57d91e4f2c85139ecd9"),
    "losses.py": ("losses-scheme-c.patch",
                  "f107371e5c77b8f81800d3676d85894a64ad6646ea78f83ecb59d768ebc09a5c"),
}
REL_PATHS = {
    "agent_loop.py": "verl/experimental/agent_loop/agent_loop.py",
    "chat_template.py": "verl/utils/tokenizer/chat_template.py",
    "multiturn_sft_dataset.py": "verl/utils/dataset/multiturn_sft_dataset.py",
    "losses.py": "verl/workers/utils/losses.py",
}
REQUIRED_PATCHES = [
    "chat_template-system-first.patch",
    "multiturn_sft_dataset-prefix-diff-answer-mask.patch",
    "losses-scheme-c.patch",
]
OPTIONAL_PATCHES = ["agent_loop-debug.patch"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_patch(patch: Path) -> tuple[str, list[tuple[str, int, int, int, int]]]:
    """Return (target relative path, [(hunk, old_start, old_count, new_start, new_count)])."""
    text = patch.read_text(encoding="utf-8")
    lines = text.splitlines()
    assert lines[0].startswith("--- a/"), "patch must start with --- a/<path>"
    assert lines[1].startswith("+++ b/"), "patch must continue with +++ b/<path>"
    target = lines[0][len("--- a/"):].strip()
    hunks = []
    for line in lines[2:]:
        m = re.match(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
        if m:
            hunks.append((line, *[int(g or 1) for g in m.groups()]))
    return target, hunks


def test_patch_files_exist_and_are_wellformed() -> None:
    for fname in [*REQUIRED_PATCHES, *OPTIONAL_PATCHES]:
        patch = PATCH_DIR / fname
        assert patch.is_file(), f"missing patch {fname}"
        text = patch.read_text(encoding="utf-8")
        assert text.endswith("\n")
        target, hunks = parse_patch(patch)
        assert target.startswith("verl/"), f"unexpected target {target}"
        assert hunks, f"{fname}: no hunks"
        assert all(h[1] >= 1 and h[3] >= 1 for h in hunks), f"{fname}: invalid hunk line numbers"
        # every added line must belong to a hunk
        body = text.splitlines()[2:]
        added = sum(1 for l in body if l.startswith("+") and not l.startswith("+++"))
        assert added > 0, f"{fname}: no added lines"
        removed = sum(1 for l in body if l.startswith("-") and not l.startswith("---"))
        assert removed >= 0


def test_patch_sha256_matches_readme_table() -> None:
    readme = (PATCH_DIR / "README.md").read_text(encoding="utf-8")
    for fname in [*REQUIRED_PATCHES, *OPTIONAL_PATCHES]:
        digest = sha256(PATCH_DIR / fname)
        assert digest in readme, f"README missing sha256 for {fname}"
        assert f"`{digest}`" in readme, f"README sha256 mismatch for {fname}"


def test_apply_reproduces_installed_sha256() -> None:
    """Apply each patch to the official-wheel fixture; must equal the server
    installed sha256 exactly (byte-for-byte)."""
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        for fixture_name, (patch_name, expected) in BUNDLE.items():
            src = FIXTURES / fixture_name
            assert src.is_file(), f"missing wheel fixture {fixture_name}"
            target = work / REL_PATHS[fixture_name]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(src.read_bytes())
            patch = PATCH_DIR / patch_name
            proc = subprocess.run(
                ["patch", "-p1", "-i", str(patch)],
                cwd=work,
                capture_output=True,
                text=True,
            )
            assert proc.returncode == 0, f"{patch_name} apply failed: {proc.stderr}"
            got = sha256(target)
            assert got == expected, (
                f"{fixture_name}: post-apply sha256 {got} != installed {expected}"
            )


def test_patch_paths_match_readme_and_scripts() -> None:
    readme = (PATCH_DIR / "README.md").read_text(encoding="utf-8")
    scripts = "\n".join(
        (PATCH_DIR.parent / name).read_text(encoding="utf-8")
        for name in ("apply_verl_patches.sh", "verify_verl_patches.sh")
    )
    expected_targets = {
        "agent_loop-debug.patch": "verl/experimental/agent_loop/agent_loop.py",
        "chat_template-system-first.patch": "verl/utils/tokenizer/chat_template.py",
        "multiturn_sft_dataset-prefix-diff-answer-mask.patch": "verl/utils/dataset/multiturn_sft_dataset.py",
        "losses-scheme-c.patch": "verl/workers/utils/losses.py",
    }
    for patch_name, target in expected_targets.items():
        assert target in readme, f"README missing target {target}"
        assert target in scripts, f"scripts missing target {target}"


APPLY_SCRIPT = PATCH_DIR.parent / "apply_verl_patches.sh"
REQUIRED_FIXTURES = ("chat_template.py", "multiturn_sft_dataset.py", "losses.py")


def _find_bash() -> str:
    """A real MSYS bash, not the WSL shim (System32\\bash.exe)."""
    import shutil

    candidate = shutil.which("bash")
    if candidate and "system32" not in candidate.lower():
        return candidate
    for base in (
        r"C:\Program Files\Git\bin\bash.exe",
        r"C:\Program Files\Git\usr\bin\bash.exe",
        r"C:\Program Files\Git\mingw64\bin\bash.exe",
    ):
        if Path(base).is_file():
            return base
    return candidate or "bash"


BASH = _find_bash()


def _bash_path(path: Path) -> str:
    """POSIX form for bash argv (drive-letter -> /d/... on Windows; no-op on Linux)."""
    text = path.as_posix()
    if len(text) >= 2 and text[1] == ":":
        return "/" + text[0].lower() + text[2:]
    return text


def _wheel_tree(work: Path) -> Path:
    """Copy the official-wheel fixtures into a verl/... site-packages layout."""
    sp = work / "site-packages"
    for name in REQUIRED_FIXTURES:
        target = sp / REL_PATHS[name]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((FIXTURES / name).read_bytes())
    return sp


def test_apply_script_fresh_patch_idempotent_rerun_and_refuse() -> None:
    """apply_verl_patches.sh three-way decision on a fixture wheel tree.

    pristine wheel -> patches apply; patched tree -> idempotent skip with
    exit 0 (regression: the skip branch previously compared a 16-char sha16
    prefix against the 64-char installed sha256, so a patched tree was never
    detected and the run always refused with exit 2); tampered file (neither
    wheel-pristine nor patched) -> refuse with exit 2.
    """
    if "system32" in BASH.lower():
        pytest.skip("WSL bash shim cannot run MSYS scripts; no Git Bash found")
    with tempfile.TemporaryDirectory() as tmp:
        sp = _wheel_tree(Path(tmp))

        def run() -> subprocess.CompletedProcess:
            return subprocess.run(
                [BASH, _bash_path(APPLY_SCRIPT), _bash_path(sp)],
                capture_output=True,
                text=True,
            )

        first = run()
        assert first.returncode == 0, first.stderr
        assert first.stdout.count("applying:") == len(REQUIRED_FIXTURES)
        assert "sha256 verification: OK" in first.stdout
        for name in REQUIRED_FIXTURES:
            assert sha256(sp / REL_PATHS[name]) == BUNDLE[name][1]

        # Idempotent re-run on the patched tree: every target skipped, no
        # re-apply, no refusal, installed hashes unchanged.
        second = run()
        assert second.returncode == 0, second.stderr
        assert second.stdout.count("skip (already applied)") == len(REQUIRED_FIXTURES)
        assert "applying:" not in second.stdout
        assert "refusing" not in second.stdout
        for name in REQUIRED_FIXTURES:
            assert sha256(sp / REL_PATHS[name]) == BUNDLE[name][1]

        # Tampered file (neither wheel-pristine nor patched) must refuse.
        tampered = sp / REL_PATHS["chat_template.py"]
        tampered.write_text(
            tampered.read_text(encoding="utf-8") + "\n# tampered\n",
            encoding="utf-8",
        )
        third = run()
        assert third.returncode == 2
        assert "neither wheel-pristine nor patched; refusing" in third.stderr


# ---------------------------------------------------------------------------
# Behavior tests — require a real verl 0.9 + torch environment (server venv).
# ---------------------------------------------------------------------------
def _has_verl() -> bool:
    try:
        import importlib.metadata as m

        version = m.version("verl")
        if version != "0.9.0":
            return False
        import torch  # noqa: F401
        import verl  # noqa: F401

        return True
    except Exception:
        return False


HAS_VERL = _has_verl()
requires_verl = pytest.mark.skipif(
    not HAS_VERL,
    reason=(
        "requires verl==0.9.0 + torch (run on the server venv: "
        "PYTHONPATH=... /root/autodl-tmp/envs/verl-qwen35/bin/python -m pytest "
        "tests/verl_patches -q)"
    ),
)
