"""Unique raw200 directory; immutable receipts and unchanged source loaders."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
R13 = ROOT.parent/'finance_teacher_coverage_trial_20261025_r13'
PREREQ = R13/'reviews/collection200_prerequisites'
sys.path.insert(0, str(R13))
import r13_common as pinned

read, put, sha, binding, exact, load = (getattr(pinned, k) for k in
                                      ('read','put','sha','binding','exact','load'))
R9, R10, R11, R12, GLOBAL = pinned.R9, pinned.R10, pinned.R11, pinned.R12, pinned.GLOBAL
sys.path.insert(0, str(PREREQ))


from release_inputs import read_config, verify_files

_original_read = read

def read(p):
    return read_config(p) if Path(p).resolve() == ROOT / 'config.json' else _original_read(p)

def verify_freeze():
    return verify_files()
