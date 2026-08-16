from pathlib import Path
import sys

# The CLI lives in the repository's script package rather than the installable src package.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
