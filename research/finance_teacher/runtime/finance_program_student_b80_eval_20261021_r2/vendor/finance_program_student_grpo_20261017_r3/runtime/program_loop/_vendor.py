"""Locate the immutable local snapshot without requiring model libraries."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor"
for path in (VENDOR, VENDOR / "runtime" / "code"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
